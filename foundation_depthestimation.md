# Depth Estimation — Deep Conceptual Notes

Companion notes for Senior CV Researcher interview prep.
Topics: monocular depth, stereo depth, self-supervised methods, evaluation metrics, SLAM connections, and more.

---

## Depth Estimation: Fundamentals — Monocular vs Stereo

---

### What Is Depth Estimation?

A camera sensor captures a 2D projection of a 3D world. Every pixel records **color/intensity** but the original **distance from the camera to that surface point** is lost in projection. Depth estimation is the task of recovering it:

```
3D world point P = (X, Y, Z)
Camera projects it to 2D pixel p = (u, v)

Depth estimation: given (u, v) → recover Z
```

The output is a **depth map** — a dense array the same size as the image, where each pixel value = distance in meters (or relative depth):

```
RGB image:              Depth map:
┌──────────────┐        ┌──────────────┐
│              │        │░░░░░░░░██████│  dark = near
│  [scene]     │   →    │░░░░░████████│
│              │        │████████████ │  bright = far
└──────────────┘        └──────────────┘

Each pixel value = Z (distance in meters from camera)
```

---

### Why Depth Is Lost in a Camera: The Projection Problem

A standard **pinhole camera** projects 3D points onto a 2D image plane via perspective projection:

$$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$

where $f_x, f_y$ = focal lengths, $(c_x, c_y)$ = principal point (image center), $(X, Y, Z)$ = 3D point.

The critical observation: **Z divides out**. Given only $(u, v)$, infinitely many 3D points $(X, Y, Z)$ map to the same pixel:

```
All these points map to the same pixel (u, v):

  Camera ──────────────────→ pixel (u,v)
            ✦ (near, small object)
                  ✦ (medium distance, medium object)
                        ✦ (far, large object)

The ray from camera through (u,v) contains all of them.
Depth estimation must figure out WHICH point on the ray is real.
```

This is the **depth ambiguity** or **scale-depth ambiguity** — the core challenge that makes depth estimation hard.

---

### Verifying and Explaining the Camera Ray Diagram

The diagram above is **correct**. Here is exactly what it means:

#### The Geometry

A pinhole camera works by projecting every 3D point onto the image plane through a single point called the **optical center**.

```
                        Image plane
                        (sensor/film)
                             │
    3D world                 │
                             │
    ✦ P₁ (near)   ──────────→│──→  pixel (u, v)
         \                   │
          ✦ P₂ (medium) ────→│──→  same pixel (u, v)
               \             │
                ✦ P₃ (far) →│──→  same pixel (u, v)
                             │
              Optical        │
              center ●───────┘
              (camera)
```

All three 3D points P₁, P₂, P₃ lie on the **same ray** — the line that starts at the optical center and passes through pixel $(u,v)$ on the image plane.

#### Why They All Map to the Same Pixel

From the projection equations, if you **double both X and Z** (move twice as far along the ray):

$$u = f_x \frac{2X}{2Z} + c_x = f_x \frac{X}{Z} + c_x \quad \text{(unchanged)}$$

The $Z$ cancels. So any point of the form $(kX, kY, kZ)$ for any scalar $k > 0$ maps to **exactly the same pixel**. All such points lie on the same ray from the optical center.

#### Concrete Example

Camera with $f = 700$px, image center $c_x = 320$:

```
Point A:  X=1m,  Z=5m    → u = 700 × (1/5)   + 320 = 460 px
Point B:  X=2m,  Z=10m   → u = 700 × (2/10)  + 320 = 460 px
Point C:  X=0.5m, Z=2.5m → u = 700 × (0.5/2.5) + 320 = 460 px

All three → pixel 460.  All lie on the same ray.
```

#### What "Depth Estimation Must Solve"

The camera records only $(u, v)$ — the endpoint of the ray. It cannot record *where along the ray* the surface was. So the depth estimator's job is:

```
Given: the ray direction (from u,v and camera intrinsics)
Find:  the parameter t such that P = camera_center + t × ray_direction
       is the actual surface point

Monocular: uses learned priors to guess t
Stereo:    uses a second camera to geometrically solve for t
```

One pixel encodes a **ray, not a point** — this is the geometric foundation of all depth estimation.

---

## Monocular Depth Estimation: One Camera, One Image

### The Fundamental Impossibility (and Why We Do It Anyway)

Mathematically, depth from a single image is **ill-posed** — there is no unique geometric solution. For any single image, infinitely many 3D scenes could have produced it.

```
These two scenes produce IDENTICAL images:

Scene A:           Scene B:
  [small chair]      [giant chair, 5× farther away]
  at 2m              at 10m

Both project to the same pixel dimensions.
A pure geometry-based solver cannot distinguish them.
```

Yet humans do it effortlessly. We use **learned priors** — visual cues that correlate with depth:

```
Monocular depth cues (what the brain / CNN uses):

1. SIZE FAMILIARITY:  "Cars are ~4m long → this car appears
                       0.1m in image → must be 40m away"

2. OCCLUSION:         "Object A covers Object B → A is closer"

3. TEXTURE GRADIENT:  "Cobblestones get denser toward horizon
                       → denser = farther"

4. LINEAR PERSPECTIVE:"Train tracks converge → convergence
                       point is far away"

5. ATMOSPHERIC HAZE:  "Distant mountains appear bluer/hazier"

6. RELATIVE HEIGHT:   "Objects lower in frame tend to be closer"

7. DEFOCUS BLUR:      "Sharp = in focus = at focal distance
                       Blurry = out of focus = near/far"

8. SHADING/SHADOW:    "Light direction + shadow length
                       encodes surface orientation + depth"
```

A deep CNN trained on large-scale data learns to recognize and combine all of these cues implicitly.

---

### Monocular Depth: What a CNN Learns

Architecture (e.g., MiDaS, DPT, Depth Anything):

```
Input: single RGB image [H × W × 3]
         ↓
   Encoder (e.g., ViT or ResNet)
   — extracts multi-scale features
   — large RF for global context (sky = far, floor = near)
         ↓
   Decoder (dense prediction head)
   — upsamples feature maps to full resolution
   — skip connections preserve fine edge detail
         ↓
Output: depth map [H × W × 1]
   — each pixel = predicted depth value
```

The encoder's large receptive field is critical — to predict depth at a pixel, the network must see the full scene context (horizon line, object relationships, scene type).

---

### Two Flavors of Monocular Depth Output

**1. Metric depth** — absolute distances in meters
```
pixel (u,v) → depth = 4.73 meters
Requires: training data with ground-truth metric depth (LiDAR)
Problem: scale ambiguity — hard to generalize across scenes
```

**2. Relative / affine-invariant depth** — ranks depth order, not absolute values
```
pixel (u,v) → depth = 0.67 (arbitrary units)
Farther pixels > closer pixels, but no metric meaning
Training: can use stereo, 3D movies, SfM — much more data available
Problem: can't use for robotics / navigation directly (no scale)
```

MiDaS and Depth Anything predict **relative depth** — trained on millions of diverse images by normalizing depth to remove scale and shift. This gives excellent generalization but no metric output.

---

### Monocular Depth: Key Limitation — Scale Ambiguity

```
Model sees: a car that subtends 100×50 pixels

Is this:
  A) a normal car at 10m?    → depth = 10m
  B) a toy car at 0.5m?      → depth = 0.5m
  C) a bus at 20m?           → depth = 20m

Without additional cues (known camera height, known object size,
or a second image), ALL THREE are geometrically consistent.
```

Scale ambiguity is why monocular metric depth is hard to generalize — a model trained on driving scenes (camera 1.5m above ground, cars 10–50m away) fails when deployed on a drone (camera 50m above ground). The learned priors are scene-specific.

---

## Stereo Depth Estimation: Two Cameras, One Geometry

### The Core Principle: Triangulation

With **two cameras** (a stereo pair) capturing the same scene simultaneously, depth becomes geometrically solvable — no learned priors needed in principle.

```
Stereo camera setup (rectified):

Left camera ●─────────────────────────────
            │←────── baseline B ─────────→│
Right camera ●─────────────────────────────

Both cameras point in the same direction.
Their image planes are coplanar (after rectification).
```

The **disparity** $d$ is the horizontal pixel difference between matching points:

$$d = u_L - u_R$$

And depth $Z$ is recovered by triangulation:

$$Z = \frac{f \cdot B}{d}$$

where $f$ = focal length (pixels), $B$ = baseline (meters), $d$ = disparity (pixels).

---

### Why This Works: Geometric Intuition

```
Close object:                   Far object:

Left cam:   │    ●              Left cam:   │         ●
            │   /                           │        /
Right cam:  │  /                Right cam:  │       /
            │ /                             │      /
            ●                               ●

Large angular difference        Small angular difference
→ large disparity d             → small disparity d
→ small depth Z (close)         → large depth Z (far)

Z = f·B/d:  as d↑, Z↓          as d↓, Z↑
```

Numerically: if $f = 700$ px, $B = 0.12$ m (typical stereo camera):

```
Object at 1m:   d = 700 × 0.12 / 1.0  = 84 px disparity
Object at 5m:   d = 700 × 0.12 / 5.0  = 16.8 px disparity
Object at 20m:  d = 700 × 0.12 / 20.0 = 4.2 px disparity
Object at 50m:  d = 700 × 0.12 / 50.0 = 1.68 px disparity ← sub-pixel!
```

This reveals the depth range limit of stereo: **at long range, disparity becomes sub-pixel** — below detector resolution — so depth accuracy degrades proportionally with $Z^2$:

$$\sigma_Z = \frac{Z^2}{f \cdot B} \sigma_d$$

where $\sigma_d$ is the disparity uncertainty (~0.5px for best algorithms). Double the distance → 4× worse depth accuracy.

---

### Stereo: The Matching Problem

After rectification, the search is constrained to the **same row** (epipolar constraint):

```
Left image pixel (u_L, v):
  Corresponding point in right image must lie on row v
  Search range: u_R ∈ [u_L - max_disp, u_L]

  ┌────────────────────────┐    ┌────────────────────────┐
  │          ●             │    │    Search →  ●         │
  │     at row 240         │ →  │    on row 240           │
  └────────────────────────┘    └────────────────────────┘
  Left image                    Right image
```

Traditional methods: compare patches (SSD, NCC, Census transform).
Deep learning methods (PSMNet, RAFT-Stereo): learn matching cost from data.

---

### Stereo Failure Modes

```
1. TEXTURELESS REGIONS:
   White wall → no texture → matching is ambiguous

2. OCCLUSIONS:
   Object visible in left camera but hidden in right
   → no correspondence exists → no depth

3. REFLECTIVE / TRANSPARENT SURFACES:
   Glass, mirror → you match the reflection, not the surface

4. LONG RANGE (d < 1px):
   Beyond ~f·B / 1px meters, accuracy degrades rapidly

5. BASELINE TOO SMALL:
   Small B → small disparity even for near objects
   Trade-off: larger B = better far-depth, worse near-depth
```

---

## Monocular vs Stereo: Side-by-Side

| Property | Monocular | Stereo |
|---|---|---|
| Hardware | Single camera | Two synchronized cameras |
| Depth principle | Learned visual priors | Geometric triangulation |
| Scale | Relative (ambiguous) unless metric training | Absolute metric depth |
| Accuracy (near range) | Moderate | High |
| Accuracy (far range) | Consistent (uses context) | Degrades as $Z^2/B$ |
| Textureless regions | Handles via global context | Fails (no matching signal) |
| Compute | One forward pass | Matching across image pair (heavier) |
| Generalization | Weaker across scene types | Strong (geometry is universal) |
| Outdoor robotics | Limited (scale unknown) | Standard (ZED, RealSense) |

---

### The Hybrid: Self-Supervised Monocular Depth (Preview)

Modern methods (Monodepth2, Depth Anything v2) train monocular networks using **stereo or video supervision** — no LiDAR labels needed:

```
Training time:  use stereo pair → compute photometric loss
                (synthesize left image from right using predicted depth)
Inference time: single image only → metric-like depth
```

This gives monocular convenience with near-metric accuracy — the best of both worlds when a stereo rig is available only during training.

---

**Interview one-liner:**
> Depth estimation recovers the Z coordinate lost during camera projection. Monocular depth uses a single image and relies on learned scene priors (texture gradients, occlusion, size familiarity) to produce relative depth — geometrically ambiguous but generalizable. Stereo depth uses two synchronized cameras and recovers metric depth via triangulation: $Z = fB/d$, where disparity $d$ shrinks with distance, making stereo accurate at short range but unreliable beyond ~$fB$ meters.

---

## Depth Ambiguity and Scale-Depth Ambiguity: The Core Challenge

---

### What "Ambiguity" Means in Mathematics

A problem is **ambiguous** when the same observation is consistent with multiple answers. There is no unique solution without additional information.

For depth estimation from a single image:

```
Observation:  pixel (u, v) has color (R=120, G=80, B=60)
Question:     what is the depth Z of the surface that produced this pixel?

Answer A:     Z = 2m  (small brown pebble up close)
Answer B:     Z = 20m (large brown rock far away)
Answer C:     Z = 200m (huge brown cliff very far)

ALL THREE are geometrically consistent with the same pixel.
```

This is the ambiguity. The image provides **no information** that rules out any of these options on purely geometric grounds.

---

### Why the Ambiguity Exists: The Projection Collapse

The pinhole camera projection compresses 3D space into 2D:

$$\text{3D point} \;(X, Y, Z) \;\xrightarrow{\text{project}} \;\text{2D pixel} \left(f\frac{X}{Z},\; f\frac{Y}{Z}\right)$$

This is a **many-to-one** mapping — an entire **ray** in 3D collapses to a single point in 2D. Specifically, every point of the form:

$$P_k = (kX,\; kY,\; kZ), \quad k > 0$$

maps to **exactly the same pixel** regardless of $k$.

```
The "ambiguity ray":

Camera
  ●──────────────────────────────────────→  (direction fixed by pixel)
        P₁ at k=0.5         P₂ at k=1        P₃ at k=3
       (X/2, Y/2, Z/2)     (X, Y, Z)        (3X, 3Y, 3Z)
         all project to the same (u, v)
```

The scalar $k$ is the **free parameter** — it encodes depth. The camera throws it away during projection.

---

### The Two Intertwined Ambiguities

The term "scale-depth ambiguity" actually contains **two entangled problems**:

---

#### Ambiguity 1: Depth vs Object Size

Given a pixel that shows a brown circle of radius 50px, which is it?

```
Case A: Small coin (r=1cm) at 20cm:
  projected radius = f × (0.01m / 0.20m) = f × 0.05

Case B: Large plate (r=10cm) at 2m:
  projected radius = f × (0.10m / 2.0m)  = f × 0.05

Case C: Enormous wheel (r=1m) at 20m:
  projected radius = f × (1.0m / 20.0m)  = f × 0.05

All three: IDENTICAL projected radius.
```

What's preserved under projection is the **angular size** $\theta = r/Z$, not the physical size $r$ or depth $Z$ independently. You can only recover the ratio — not either quantity alone.

This is the **size-distance ambiguity**: object size and depth trade off perfectly. A 2× larger object at 2× the distance is identical in the image.

---

#### Ambiguity 2: Absolute Scale (Scene Scale Unknown)

Even if you know the **shape** of a scene perfectly — its relative geometry — you don't know its **absolute scale**.

```
Imagine you perfectly reconstructed a 3D scene from images:

"Scene A": room is 3m × 4m × 2.5m (width × depth × height)

Is this:
  A model house (0.3m × 0.4m × 0.25m)?   ← scale k=0.1
  A real room (3m × 4m × 2.5m)?           ← scale k=1
  A warehouse? (30m × 40m × 25m)?          ← scale k=10

You CANNOT tell from images alone. All three project identically
if you scale the camera baseline by the same factor k.
```

This is the **global scale ambiguity** — the entire 3D reconstruction is defined only up to a multiplicative constant. This is why Structure-from-Motion (SfM) and SLAM produce reconstructions in "arbitrary units" unless a metric reference is provided (GPS, known object size, IMU).

---

### How They Combine in Practice

Consider a monocular depth network predicting the depth of a car:

```
Network sees: car occupying 200×80 pixels in a 1280×720 image

The network's reasoning:
  "This looks like a car (learned prior: cars ≈ 4.5m long)"
  "It subtends 200/1280 = 15.6% of image width"
  "Focal length ≈ 700px (assumed)"
  "Z ≈ f × (car_length / projected_length)"
  "Z ≈ 700 × (4.5 / 200) ≈ 15.75m"
```

But this reasoning has hidden assumptions:
1. It IS a car, not a toy car or a bus → **object identity assumption**
2. The focal length is known → **camera calibration assumption**
3. Real cars are ~4.5m long → **world knowledge prior**

If ANY of these is wrong, the depth estimate is wrong. There is no geometric fallback.

---

### Why This Doesn't Happen with Stereo

Stereo **breaks** the ambiguity because you have two viewpoints separated by a known distance $B$:

```
Left camera                  Right camera
     ●─────────────────────────●
              B = 0.12m

Point P projects to:
  Left:  u_L = f·X/Z + cx
  Right: u_R = f·(X-B)/Z + cx

Disparity: d = u_L - u_R = f·B/Z

Solve for Z:  Z = f·B/d
```

The baseline $B$ **injects absolute metric scale**. The free parameter $k$ is no longer free — moving the point along the ray would change $d$ in a predictable way that contradicts the actual measurement.

```
Can Z = 2×(f·B/d) be correct?
  That would require d' = d/2 (half the disparity)
  But we MEASURED d in the right image — it's d, not d/2.
  Contradiction. ✗

Therefore Z = f·B/d is the UNIQUE solution.
```

The ambiguity is eliminated geometrically, with no priors needed.

---

### The Formal Statement: What Breaks Scale

Scale ambiguity is broken by any of the following:

| What breaks it | How | Used in |
|---|---|---|
| Second camera (stereo) | Baseline injects metric scale | Stereo depth, SLAM |
| Known object size | "This is a 4.5m car" | Object-based depth |
| Known camera height | "Camera is 1.5m above ground" | Autonomous driving |
| IMU / accelerometer | Measures real-world acceleration | Visual-inertial odometry |
| LiDAR fusion | Sparse metric depth anchors | Semi-supervised depth |
| Known focal length + object | $Z = f \cdot \text{size} / \text{pixels}$ | AR marker tracking |

Without at least one of these, depth is **scale-ambiguous** from a single image.

---

### What Monocular Networks Actually Learn

Since they can't resolve scale geometrically, monocular networks learn a **conditional distribution**:

$$P(Z \mid \text{image}) = \text{learned from training distribution}$$

The network memorizes: "scenes that look like this typically have these depths." This works well **within the training distribution** but fails when:

```
Training: dashcam video at 1.5m height, highway (Z range: 5–100m)
Test:     drone footage at 50m height, city (Z range: 10–500m)

The learned priors (road size, car size, building proportions)
are completely different → depth predictions collapse.
```

---

### The Affine-Invariant Trick (MiDaS, Depth Anything)

Modern monocular models sidestep scale ambiguity by **not predicting metric depth at all**. They predict depth up to an unknown scale $s$ and shift $t$:

$$\hat{d} = s \cdot Z + t$$

Training loss is computed after normalizing out $s$ and $t$ from the ground truth:

$$\mathcal{L} = \left\| \frac{\hat{d} - \text{median}(\hat{d})}{\text{MAD}(\hat{d})} - \frac{d_{gt} - \text{median}(d_{gt})}{\text{MAD}(d_{gt})} \right\|$$

This makes the loss **invariant to scale and shift** — the network only has to get the **relative ordering and shape** of the depth map correct. Result: can train on wildly diverse datasets with different scale conventions, and the model generalizes across scenes. But you lose the ability to directly output meters.

---

**Interview one-liner:**
> Scale-depth ambiguity arises because camera projection is a many-to-one mapping: the entire ray $P_k = k(X,Y,Z)$ collapses to one pixel, making depth and object size interchangeable with no geometric way to distinguish them from a single view. Stereo breaks this by measuring the same point from two known positions — the baseline injects absolute metric scale. Monocular networks work around it by learning scene priors, but this ties them to training distributions and prevents true metric generalization.

---

## Monocular Depth: Single Image → Depth Map — The Underdetermined Problem

---

### What "Underdetermined" Means

A system of equations is **underdetermined** when there are more unknowns than constraints — infinitely many solutions satisfy all the equations simultaneously.

For monocular depth:

```
Unknowns:    depth Z(u,v) for every pixel → H × W unknowns
             (e.g., 480 × 640 = 307,200 unknowns)

Constraints: pixel color I(u,v) for every pixel → H × W equations

On the surface: same number of equations and unknowns!
But the equations DON'T constrain depth:

  I(u,v) = surface_color(X, Y, Z) × lighting(X, Y, Z)

  This depends on: material color, surface normal, light position,
  AND depth Z — all unknown simultaneously.

  One equation, multiple unknowns → severely underdetermined.
```

Even worse: changing depth while scaling the scene leaves the image **completely unchanged**. It's not just underdetermined — it's **fundamentally degenerate**.

---

### Formal Proof: Infinitely Many 3D Scenes → Same Image

Let scene $\mathcal{S}$ consist of surfaces with positions and colors. Camera with focal length $f$ produces image $\mathcal{I}$.

Now construct scene $\mathcal{S}_k$ by scaling: every 3D point $(X,Y,Z)$ → $(kX, kY, kZ)$, surface normals keep the same direction, light sources scaled similarly.

Projection of $\mathcal{S}_k$:

$$u = f \frac{kX}{kZ} = f \frac{X}{Z} \quad \text{(unchanged)}$$

The $k$ cancels. The image is identical. This holds for **any** $k > 0$.

```
Scene at scale k=1:     Scene at scale k=2:     Scene at scale k=10:
  house 10m away          house 20m away          house 100m away
  (real size)             (2× bigger house)       (10× bigger house)

All three produce:         IDENTICAL image
```

This isn't a numerical coincidence — it's a structural property of perspective projection. No amount of image analysis can distinguish them without external information.

---

### The Degrees of Freedom Problem

```
3D scene:
  Each surface point has position (X, Y, Z) + material properties
  → infinite-dimensional space of possible scenes

Single image:
  H × W pixels, each with 3 color values
  → H × W × 3 real numbers of information

Mapping: scene → image is infinitely many-to-one
  The "fiber" above each image (all scenes that produce it)
  is infinitely large
```

In information theory terms: the image has $H \times W \times 3 \times 8$ bits. The 3D scene has infinitely more degrees of freedom. No injective inverse exists.

---

### What "Requires Learning Priors" Actually Means

Since geometry alone can't solve the problem, depth estimation relies on **statistical regularities** — patterns that are true of real-world scenes **on average** even if not always.

A **prior** is a probability distribution over depth given image features:

$$P(Z \mid \mathcal{I}) = \frac{P(\mathcal{I} \mid Z) \cdot P(Z)}{P(\mathcal{I})}$$

In Bayesian terms:
- $P(Z)$ = **prior**: what depths are likely in the world before seeing the image
- $P(\mathcal{I} \mid Z)$ = **likelihood**: how likely is this image given this depth
- $P(Z \mid \mathcal{I})$ = **posterior**: best depth estimate given the image

A deep neural network implicitly learns to approximate this posterior from millions of (image, depth) pairs.

---

### The Priors Being Learned: Concrete Examples

#### Prior 1: Vertical Position → Depth

```
In road/outdoor scenes:
  Objects near bottom of image → close (on the ground, near camera)
  Objects near horizon → far
  Sky (top of image) → infinitely far

Network learns: "pixel at row 420/480 is probably road at ~3m"
                "pixel at row 200/480 is probably sky or distant object"

Fails in:
  - Indoor scenes (ceiling is far, floor is close)
  - Aerial photography (everything is "below" camera)
  - Scenes with ramps or hills (ground is not flat)
```

#### Prior 2: Texture Gradient → Depth

```
Uniform texture (grass, cobblestones, sand):
  Texture elements appear SMALLER and MORE DENSELY PACKED
  as distance increases

  ● ●●●●●●●●●●●●●●●●  ← near: sparse, large texture elements
  ●●●●●●●●●●●●●●●●●● ← medium
  ················    ← far: dense, tiny, eventually blur together

Failure case: textureless surface (white wall)
              → no gradient signal → depth uncertain
```

#### Prior 3: Occlusion Ordering

```
Object A occludes object B at their shared boundary:
  → A is closer than B (definitive ordering, no scale needed)

  ┌─────────┐
  │    A    │──── boundary pixel: A is in front of B
  └─────────┘
     │   ┌──────────────────┐
     │   │        B         │
     │   └──────────────────┘
```

#### Prior 4: Familiar Object Size (Absolute Scale Prior)

```
Trained on millions of images with:
  - Cars → typically 1.5m tall, 4.5m long
  - People → typically 1.7m tall
  - Doors → typically 2.0m tall

Network sees a person occupying 100px height in 720px image:
  "Person is ~1.7m. Appears 100/720 = 13.9% of frame height.
   At f=700px: Z ≈ 700 × 1.7 / 100 = ~12m"

Strongest prior for metric depth — but requires correct object recognition.
```

#### Prior 5: Atmospheric Perspective

```
Near objects:  high contrast, saturated colors, sharp edges
Far objects:   low contrast, desaturated/bluish, hazy edges

Network learns: "hazy, low-contrast region → far away"

Failure case: foggy day → everything looks "far" even if close
```

---

### Why Deep Networks Are Needed (Not Just Rules)

Priors **conflict and interact**:

```
Scenario: a person standing in front of a white wall

Prior 1 (vertical position):  "person is at row 300 → ~5m"
Prior 3 (occlusion):          "person occludes wall → person is closer"
Prior 4 (familiar size):      "person is 80px tall → ~15m"

These give different estimates. How do you combine them?
```

A deep network learns:
1. Which priors to trust in which contexts
2. How to weight conflicting priors
3. When to fall back to relative depth (give up on metric)
4. Non-obvious correlations humans haven't explicitly identified

This requires millions of training examples to cover the enormous diversity of real scenes.

---

### The Underdetermination Manifests as Training Instability

Training a naive monocular depth network with an L1 or L2 loss often fails because:

```
For the same input patch:
  Training example 1: road at 5m  (depth=5)
  Training example 2: road at 15m (same visual appearance, depth=15)

The network sees the same image → must predict different depths
→ loss gradient pulls in opposite directions
→ network averages to ~10m for both → wrong for both

This is the "regression to the mean" problem:
  the network predicts blurry, averaged depth maps
  because per-pixel depth is poorly constrained by appearance alone
```

Solutions that combat underdetermination during training:

```
1. SCALE-INVARIANT LOSS (Eigen et al.):
   Compute loss on log-depth differences, not absolute depths
   → invariant to global scale factor

2. ORDINAL/RANKING LOSS:
   "Is pixel A closer than pixel B?" (binary)
   → more constrained than exact depth, uses more training pairs

3. RELATIVE DEPTH NORMALIZATION (MiDaS):
   Normalize depth to [0,1] per image during training
   → removes scale and shift, focuses on shape

4. MULTI-TASK LEARNING:
   Jointly predict depth + surface normals + semantic labels
   → normals constrain depth gradients, semantics constrain scale
```

---

### The Prior Generalization Problem

Because monocular depth is fundamentally about learned priors, it suffers from **distribution shift**:

```
KITTI (outdoor driving):          NYUv2 (indoor):
  Camera at 1.5m height             Camera handheld at ~1.2m
  Depth range: 1–80m                Depth range: 0.5–10m
  Dominant colors: grey/green       Dominant colors: varied

Model trained on KITTI, tested on NYUv2:
  → "floor" predicted as ~30m (was trained: horizontal surface = road = 30m)
  → "ceiling" predicted as sky depth
  → total failure

The priors learned for one domain ACTIVELY HARM performance in another.
```

This is the core reason why **relative depth models** (MiDaS, DPT, Depth Anything) exist — by learning scale-invariant relative depth across many datasets simultaneously, they build domain-agnostic priors.

---

### Summary: Why It's Hard

```
Problem source:         Consequence:
──────────────          ────────────
Many-to-one projection  No unique geometric solution exists
Scale-depth coupling    Absolute metric depth requires external reference
Prior conflict          Depth estimates inconsistent across scene regions
Distribution shift      Learned priors don't transfer across domains
Textureless regions     Core visual cues (texture gradient) unavailable
Thin structures         Small objects have few pixels → high uncertainty
```

---

**Interview one-liner:**
> Monocular depth is underdetermined because perspective projection is a many-to-one mapping — infinitely many 3D scenes differing only by a global scale factor produce identical images. A depth network can't solve this geometrically, so it instead learns statistical priors from training data: vertical position correlates with depth, texture gradients indicate distance, familiar object sizes anchor metric scale. These priors work well within the training distribution but generalize poorly, which is why metric monocular depth remains an open challenge while relative depth (predicting depth ordering, not absolute values) is largely solved.

---

*End of notes — continued in next session.*