# Object Detection & Tracking — Deep Foundational Notes
*Interview prep for Senior Computer Vision Researcher roles*

---

## Q: What do you mean by "Localization accuracy: bounding box must be precise"?

### What is a Bounding Box?
A **bounding box** is a rectangle drawn around an object in an image. It's defined by 4 numbers:

$$[x_{min}, y_{min}, x_{max}, y_{max}]$$

or equivalently: center $(c_x, c_y)$, width $w$, height $h$.

---

### What does "precise" mean here?

The model must predict a box that **tightly wraps** the object — not too big, not too small, not shifted.

```
Too loose (bad):          Tight (good):         Shifted (bad):
┌──────────────┐          ┌──────┐              ┌──────┐
│              │          │ 🐱  │              │      │ 🐱
│   🐱         │          └──────┘              └──────┘
│              │
└──────────────┘
```

---

### Why does this matter?

**IoU (Intersection over Union)** is how we measure box precision:

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

- IoU = 1.0 → perfect box
- IoU = 0.5 → box is accepted in PASCAL VOC standard
- IoU = 0.75 → stricter MS-COCO standard

Even if the model **correctly says "there is a cat"**, if the box is sloppy, the detection is counted as **a miss** (False Negative) at high IoU thresholds.

---

### Why is it hard?

1. **Objects have irregular shapes** — a tight box around a giraffe's neck vs. its body is ambiguous
2. **Objects are at different scales** — tiny pedestrian far away vs. car nearby
3. **Occlusion** — part of the object is hidden, where does the box end?
4. **Regression is continuous** — predicting exact pixel coordinates is harder than a yes/no classification

---

### The two tasks combined

Object detection = **classification + localization** together:

| Task | Output | Loss |
|------|--------|------|
| Classification | "Is it a cat?" | Cross-entropy |
| Localization | "Where exactly?" | Regression loss (L1/L2/IoU loss) |

A model must get **both right** to score a true positive.

---

**Interview-ready one-liner:**
> "Localization accuracy means the predicted bounding box must tightly align with the ground truth object boundary, measured by IoU — because even a correct class label is useless if the box doesn't reliably locate the object for downstream tasks like cropping, tracking, or segmentation."

---

## Q: Explain IoU (Intersection over Union) in detail

### The Core Idea

You have two boxes:
- **Ground Truth (GT)** — the human-drawn "correct" box
- **Predicted (P)** — what your model outputs

IoU asks: **"How much do they overlap relative to their combined area?"**

$$\text{IoU} = \frac{|GT \cap P|}{|GT \cup P|}$$

---

### Visual Breakdown

```
Ground Truth box (GT):         Predicted box (P):
┌─────────┐                    
│         │        ┌─────────┐
│   GT    │        │    P    │
│      ┌──┼────────┤         │
│      │██│  INTER │         │
└──────┼──┘        └─────────┘
       └───────────┘

██ = Intersection (overlap area)
Everything covered by either box = Union
```

$$\text{IoU} = \frac{\text{Intersection Area}}{\text{Union Area}} = \frac{\text{Intersection}}{\text{GT Area} + \text{P Area} - \text{Intersection}}$$

---

### Computing It Step by Step

Given:
- GT box: $(x_1^{gt}, y_1^{gt}, x_2^{gt}, y_2^{gt})$
- Predicted box: $(x_1^p, y_1^p, x_2^p, y_2^p)$

**Step 1: Find intersection rectangle**

$$x_1^{inter} = \max(x_1^{gt},\ x_1^p)$$
$$y_1^{inter} = \max(y_1^{gt},\ y_1^p)$$
$$x_2^{inter} = \min(x_2^{gt},\ x_2^p)$$
$$y_2^{inter} = \min(y_2^{gt},\ y_2^p)$$

**Step 2: Compute intersection area**

$$\text{Inter} = \max(0,\ x_2^{inter} - x_1^{inter}) \times \max(0,\ y_2^{inter} - y_1^{inter})$$

The $\max(0, \cdot)$ handles the case where boxes **don't overlap at all** → IoU = 0.

**Step 3: Compute union**

$$\text{Union} = \text{Area}_{GT} + \text{Area}_{P} - \text{Inter}$$

**Step 4:**

$$\text{IoU} = \frac{\text{Inter}}{\text{Union}}$$

---

### IoU Range and What Values Mean

| IoU | Meaning |
|-----|---------|
| **1.0** | Perfect overlap — identical boxes |
| **0.7 – 0.9** | Very good prediction |
| **0.5** | PASCAL VOC threshold — accepted as TP |
| **0.75** | MS-COCO strict threshold |
| **0.0** | No overlap at all |

---

### How IoU is Used in Evaluation

At inference time, for each predicted box:

1. Compute IoU with every GT box of the same class
2. If **IoU ≥ threshold** → **True Positive (TP)**
3. If **IoU < threshold** → **False Positive (FP)**
4. GT box not matched by any prediction → **False Negative (FN)**

This feeds into **Precision**, **Recall**, and ultimately **mAP**.

---

### Beyond Basic IoU — Variants Used in Modern Detectors

| Variant | What it adds | Why |
|---------|-------------|-----|
| **GIoU** | Penalizes non-overlapping boxes using enclosing box | Smooth gradient even when boxes don't overlap |
| **DIoU** | Adds center-point distance penalty | Pushes predicted center toward GT center |
| **CIoU** | DIoU + aspect ratio penalty | Used in YOLOv4/v5 loss function |

Basic IoU as a **loss** has a problem: if two boxes don't overlap, the gradient is **zero** — the model gets no learning signal. GIoU/DIoU/CIoU fix this.

---

**Interview-ready one-liner:**
> "IoU is the ratio of the overlap area to the combined area of two boxes. It's the universal metric for bounding box quality — used as an evaluation threshold (TP/FP classification), as a training loss (via differentiable variants like CIoU), and as the backbone of NMS to suppress duplicate detections."

---

## Q: Why is localization hard? — Objects have irregular shapes (giraffe example)

### The Core Problem

A bounding box is always a **rectangle**. But real objects are **not rectangles**. So you're forcing an irregular shape into a rigid geometric container — and there's no single "correct" way to do that.

---

### The Giraffe Example — Visualized

```
Real giraffe shape:        Bounding box option 1:     Bounding box option 2:
                           (tight around whole body)   (tight around visible body)

    ( head )               ┌──────────────┐            
       |                   │   ( head )   │               ( head )
       | ← long neck       │      |       │                  |
       |                   │      |       │            ┌─────┤
  [  body  ]               │  [  body  ] │            │[body│
    /     \                └──────────────┘            └─────┘
```

**Both boxes are "correct"** — but they give different $(x, y, w, h)$ values. The model and the annotator may not agree on which one to use.

---

### Three Specific Ambiguities

**1. Where does the object "end"?**

A giraffe's neck is part of the giraffe. But a tight box around just the neck gives a very different box than one around the whole animal. The **semantic boundary** (what is "the giraffe"?) is clear, but the **spatial boundary** (where does it end in pixel space?) depends on the annotator's decision.

**2. Non-convex shapes waste box area**

A giraffe's silhouette is long and thin at the top (neck), wide at the bottom (body). Any bounding box will include **large empty regions** (background pixels inside the box). This:
- Makes IoU artificially low even for good predictions
- Forces the model to regress to a box that "averages out" the shape
- Makes downstream tasks (e.g., cropping for re-id) noisy

```
┌──────────────┐
│ background   │  ← these empty corners are INSIDE the box
│   ( neck )   │
│              │
│   [  body  ] │
└──────────────┘
```

**3. Annotator disagreement introduces label noise**

Studies show that for irregular objects, IoU between two human annotators labeling the same object is often **only 0.7–0.85** — not 1.0. So the ground truth itself has noise. The model is trying to regress to an ambiguous target.

---

### Why This Matters for the Model

The model outputs $(c_x, c_y, w, h)$ as **a single deterministic answer**. But for a giraffe, many valid boxes exist. The loss function (L1, L2, CIoU) penalizes deviation from ONE ground truth box — even though slightly different boxes could be equally valid.

This is why some modern approaches moved toward:
- **Instance segmentation** (Mask R-CNN) — predict a pixel mask, not just a box
- **Probabilistic detection** — predict a distribution over box coordinates, not a point estimate

---

**Interview-ready one-liner:**
> "Bounding boxes are rectangles forced onto non-rectangular objects. For irregular shapes like a giraffe, there's no unique 'correct' box — the neck and body create ambiguity in height/width boundaries, annotators disagree, and the model must regress to one noisy target, which is why pixel-level segmentation is often preferred when precise localization matters."

---

## Q: Beyond Basic IoU — Variants Used in Modern Detectors

### Why Basic IoU Fails as a Loss Function

When you use IoU directly as a training loss, it has one critical flaw:

**If two boxes don't overlap at all → IoU = 0 → gradient = 0 → model learns nothing.**

```
GT box:       ┌──────┐
              │  GT  │
              └──────┘
                          ┌──────┐
                          │  P   │   ← Predicted box far away
                          └──────┘

IoU = 0.  No overlap.  Loss = 0.  Gradient = 0.  Model stuck.
```

---

### GIoU — Generalized IoU (2019)

**Problem it solves:** Zero gradient when boxes don't overlap.

**Key idea:** Find the **smallest enclosing box** $C$ that contains **both** GT and P. Penalize how much of $C$ is wasted.

$$\text{GIoU} = \text{IoU} - \frac{|C \setminus (GT \cup P)|}{|C|}$$

```
        ┌──────────────────────────┐  ← Enclosing box C (covers BOTH)
        │                          │
        │ ┌──────┐      ┌──────┐   │
        │ │  GT  │      │  P   │   │
        │ └──────┘      └──────┘   │
        │                          │
        │   ↑ wasted space in C    │
        │   (not covered by GT∪P)  │
        └──────────────────────────┘
```

- The **gap between GT and P** inside C is the wasted area
- Bigger gap → larger wasted area → more negative GIoU → stronger gradient pushing P toward GT
- Even when IoU = 0, the wasted-area penalty gives a **non-zero gradient**
- **Range:** $-1 \leq \text{GIoU} \leq 1$

**Limitation:** Doesn't directly minimize *how far apart* the centers are.

---

### DIoU — Distance IoU (2020)

**Problem it solves:** GIoU converges slowly — it only looks at wasted area, not center alignment.

**Key idea:** Add a penalty for the **distance between box centers**.

$$\text{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}$$

Where:
- $\rho^2(b, b^{gt})$ = squared Euclidean distance between predicted center and GT center
- $c^2$ = squared diagonal of the enclosing box C (for normalization)

```
        ┌──────────────────────────┐  ← Enclosing box C (diagonal = c)
        │                          │
        │  ┌──────┐    ┌──────┐    │
        │  │  GT  │    │  P   │    │
        │  │  ★───┼────┼──►★  │    │
        │  └──────┘    └──────┘    │
        │       ρ = center dist    │
        └──────────────────────────┘
```

- **Directly optimizes center alignment** — faster convergence than GIoU
- Penalty is normalized by $c^2$ so it's scale-invariant
- Also used in **DIoU-NMS**: suppress overlapping boxes by center distance

**Limitation:** Doesn't care about aspect ratio — a wide flat box and a tall thin box with same center distance are treated equally.

---

### CIoU — Complete IoU (2020)

**Problem it solves:** DIoU ignores shape — aspect ratio mismatch goes unpunished.

**Key idea:** Add an **aspect ratio consistency term** on top of DIoU.

$$\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v$$

Where:

$$v = \frac{4}{\pi^2} \left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$$

$$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

```
GT box (tall):   Predicted box (wide):
┌───┐            ┌────────────┐
│   │            │            │
│ ★ │     vs     │     ★      │  ← same center
│   │            └────────────┘
└───┘
Same center, same area — but wrong shape.
CIoU penalizes this. DIoU does not.
```

**CIoU optimizes three things simultaneously:**
1. **Overlap area** (IoU term)
2. **Center distance** (DIoU term)
3. **Aspect ratio** (v term)

Used in **YOLOv4, YOLOv5, YOLOv7** as the bounding box regression loss.

---

### Summary Table

| Metric | Adds | Fixes | Used in |
|--------|------|-------|---------|
| **IoU** | — | Baseline overlap | Evaluation |
| **GIoU** | Enclosing box waste penalty | Zero gradient when no overlap | Early YOLO variants |
| **DIoU** | Center distance penalty | Slow convergence of GIoU | DIoU-NMS |
| **CIoU** | Center distance + aspect ratio | Shape mismatch ignored by DIoU | YOLOv4/v5/v7 loss |

---

**Interview-ready one-liner:**
> "Basic IoU gives zero gradient when boxes don't overlap, so modern detectors use differentiable variants: GIoU adds an enclosing-box penalty for non-overlapping cases, DIoU adds center-distance minimization for faster convergence, and CIoU additionally penalizes aspect ratio mismatch — making it the most complete loss for bounding box regression, used in YOLO variants."

---

## Q: GIoU — Why Zero Gradient is a Problem and How GIoU Fixes It

### First, Understand What "Gradient" Means in Training

During training, the model predicts box coordinates $(c_x, c_y, w, h)$. The loss function tells the model **"how wrong you are"**, and the gradient tells the model **"which direction to move"** to reduce that error.

$$\text{gradient} = \frac{\partial \text{Loss}}{\partial \text{predicted coordinates}}$$

If gradient = 0 → **model receives no update → weights don't change → model is stuck.**

---

### Why Basic IoU Has Zero Gradient When Boxes Don't Overlap

Let's say GT box is at position $[10, 10, 50, 50]$ and predicted box is at $[200, 200, 240, 240]$ — completely far apart.

**Step 1: Compute intersection**

$$\text{Inter} = \max(0,\ x_2^{inter} - x_1^{inter}) \times \max(0,\ y_2^{inter} - y_1^{inter})$$

Since boxes are far apart:

$$x_2^{inter} - x_1^{inter} < 0 \Rightarrow \max(0, \text{negative}) = 0$$

$$\text{Inter} = 0$$

**Step 2: Compute IoU**

$$\text{IoU} = \frac{0}{\text{Union}} = 0$$

**Step 3: The gradient problem**

$$\frac{\partial \text{IoU}}{\partial (c_x, c_y, w, h)} = \frac{\partial}{\partial \text{coords}}\left(\frac{0}{\text{Union}}\right) = 0$$

The loss is a **flat plateau** — like the bottom of a valley with no slope:

```
Loss
│
│  ████████████████          ← flat plateau (IoU=0 everywhere boxes don't overlap)
│                  ████
│                      ████
│                          ████  ← slope only appears when boxes start overlapping
└─────────────────────────────────► box position
        far away     overlapping
```

The model has **no idea which direction to move** the predicted box. It could be off by 10 pixels or 1000 pixels — the loss is the same: 0.

---

### How GIoU Fixes This

GIoU adds the **enclosing box term** that is always computable — even when boxes don't overlap:

$$\text{GIoU} = \text{IoU} - \frac{|C \setminus (GT \cup P)|}{|C|}$$

The second term $\frac{|C \setminus (GT \cup P)|}{|C|}$ depends on the **coordinates of the predicted box** — so its gradient is **never zero**.

```
Loss (GIoU-based)
│
│  \                        ← slope even when boxes are far apart
│   \
│    \
│     \____                 ← gets flatter as boxes approach
│          \____
│               \___________  ← near-zero when boxes overlap perfectly
└─────────────────────────────► box position
        far away     overlapping
```

**Why does the enclosing box term always have a gradient?**

The enclosing box $C$ is defined by:

$$x_1^C = \min(x_1^{gt}, x_1^p), \quad x_2^C = \max(x_2^{gt}, x_2^p)$$

These are direct functions of the **predicted coordinates**. So:

$$\frac{\partial |C|}{\partial x_1^p} \neq 0, \quad \frac{\partial |C|}{\partial x_2^p} \neq 0$$

As the predicted box moves away from GT, $|C|$ (enclosing area) gets **bigger**, so the penalty $\frac{|C \setminus (GT \cup P)|}{|C|}$ grows → GIoU becomes more negative → loss increases → **gradient pushes P toward GT**.

---

### Concrete Numerical Example

| Scenario | IoU | Enclosing penalty | GIoU |
|----------|-----|-------------------|------|
| Boxes identical | 1.0 | 0 | **1.0** |
| Boxes slightly overlapping | 0.3 | 0.1 | **0.2** |
| Boxes touching (no overlap) | 0.0 | 0.3 | **-0.3** |
| Boxes far apart | 0.0 | 0.9 | **-0.9** |

Notice: as boxes move further apart, GIoU gets **more negative** — that negative value IS the gradient signal that says "you are very wrong, move the box."

---

**Interview-ready one-liner:**
> "Basic IoU is zero for all non-overlapping boxes regardless of distance, creating a flat loss surface with no gradient — the model can't learn. GIoU fixes this by subtracting a penalty based on the wasted area inside the smallest enclosing box, which is always a function of predicted coordinates and therefore always produces a non-zero gradient to guide the prediction toward the ground truth."

---

## Q: CIoU — Why Aspect Ratio Matters and How CIoU Fixes It

### First, What is Aspect Ratio?

Aspect ratio is simply **width divided by height** of a box:

$$\text{aspect ratio} = \frac{w}{h}$$

| Box | Width | Height | Aspect Ratio |
|-----|-------|--------|-------------|
| Tall person | 50px | 150px | 0.33 (tall/thin) |
| Car (side view) | 200px | 80px | 2.5 (wide/flat) |
| Square object | 100px | 100px | 1.0 |

---

### The Problem: DIoU is Blind to Shape

DIoU only cares about **center distance** and **overlap area**. Consider this scenario:

```
GT box (tall person):    Predicted box (wide/flat):

┌───┐                    ┌────────────────┐
│   │                    │                │
│   │       vs           │                │
│   │                    └────────────────┘
│   │
└───┘

Both boxes: same center ★, same area
DIoU penalty = 0  (centers are identical!)
DIoU says: "Perfect prediction!" ← WRONG
```

The predicted box has the **completely wrong shape** — it would be a terrible detection of a person — but DIoU gives it a perfect score because centers align and there's overlap.

---

### Why Shape Matters in Practice

**Example: Pedestrian detection**

A pedestrian is tall and thin. If your model predicts a wide flat box:
- The box captures the torso but misses the head and legs
- For re-identification (tracking), the crop is wrong — you're feeding the wrong pixels to the feature extractor
- For pose estimation downstream, joint locations will be wrong

```
Correct box:    Bad prediction:
┌──┐            ┌────────────────┐
│👤│     vs     │      👤        │  ← person squeezed into wrong box
│  │            └────────────────┘
│  │
└──┘
IoU might still be decent, but the box is useless.
```

---

### How CIoU Adds Aspect Ratio Penalty

CIoU formula:

$$\text{CIoU} = \text{IoU} - \underbrace{\frac{\rho^2(b, b^{gt})}{c^2}}_{\text{center distance}} - \underbrace{\alpha v}_{\text{aspect ratio}}$$

**The $v$ term** measures how different the aspect ratios are:

$$v = \frac{4}{\pi^2} \left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$$

**Why arctan?** Because we want to compare the *angle* of the width/height ratio, not the raw values. A box of $200 \times 100$ and $20 \times 10$ have the same aspect ratio — arctan captures this correctly regardless of scale.

$$\arctan\frac{w}{h} = \text{angle whose tangent is } \frac{w}{h}$$

```
arctan(w/h):

  tall box (w<h):  arctan → small angle (near 0°)
  square (w=h):    arctan → 45°
  wide box (w>h):  arctan → large angle (near 90°)

If GT is tall and P is wide:
  arctan(w_gt/h_gt) ≈ 20°
  arctan(w_p/h_p)   ≈ 70°
  difference = 50° → v is large → big penalty
```

The $\frac{4}{\pi^2}$ factor normalizes $v$ to $[0, 1]$.

---

### The $\alpha$ Weight — Balancing the Penalty

$$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

- When **IoU is low** (boxes barely overlap): $\alpha$ is small → aspect ratio matters less, focus on getting boxes to overlap first
- When **IoU is high** (boxes nearly overlap): $\alpha$ is large → now shape refinement becomes the priority

This is a smart **curriculum**: fix position first, then refine shape.

---

### Step-by-Step Numerical Example

GT box: $w=50, h=150$ (tall person) → $\arctan(50/150) = \arctan(0.33) \approx 18.4°$

Predicted box: $w=150, h=50$ (wide flat) → $\arctan(150/50) = \arctan(3.0) \approx 71.6°$

$$v = \frac{4}{\pi^2}(18.4° - 71.6°)^2 \approx \frac{4}{9.87} \times 2809 \approx 1.14 \text{ (large penalty)}$$

Compared to a good prediction: $w=55, h=145$ → $\arctan(55/145) \approx 20.8°$

$$v = \frac{4}{\pi^2}(18.4° - 20.8°)^2 \approx \frac{4}{9.87} \times 5.76 \approx 0.0023 \text{ (tiny penalty)}$$

---

### Summary: What Each Term Fixes

```
IoU alone:    ✅ overlap        ❌ no gradient if no overlap
GIoU adds:    ✅ overlap        ✅ gradient when no overlap    ❌ slow convergence
DIoU adds:    ✅ overlap        ✅ gradient                    ✅ center alignment   ❌ ignores shape
CIoU adds:    ✅ overlap        ✅ gradient                    ✅ center alignment   ✅ aspect ratio
```

---

**Interview-ready one-liner:**
> "DIoU ignores aspect ratio — two boxes with the same center and same area but different shapes get the same score. CIoU fixes this by adding a term $\alpha v$ that measures the angular difference between the width/height ratios using arctan, penalizing shape mismatch while using an adaptive weight $\alpha$ that prioritizes position alignment first and shape refinement once boxes are close — making CIoU the most complete bounding box loss."

---

## Q: Variable Number of Objects — A Core Challenge in Object Detection

### The Fundamental Problem

In **image classification**, the output is always exactly **one label**:

```
Input image → CNN → [dog: 0.95, cat: 0.03, car: 0.02]
                     ↑ always one fixed-size vector
```

In **object detection**, the output is a **list of variable length** — and you don't know that length in advance:

```
Image 1 (empty street):   → []                         ← 0 objects
Image 2 (one person):     → [(person, box)]             ← 1 object
Image 3 (crowd scene):    → [(person,box), (person,box), (car,box), ...]  ← many objects
```

**The neural network always outputs a fixed-size tensor.** So how do you map a fixed output → variable number of detections? This is the core design challenge.

---

### Why Fixed-Output Networks Struggle

A standard CNN outputs a fixed vector, e.g., shape $[1000]$ for 1000 classes. You can't just do that for detection because:

- You don't know if there are 2 objects or 200 objects
- You can't have 200 output heads — that would be a different network for every image
- Simply outputting "up to N boxes" wastes computation and requires deciding N upfront

---

### How Modern Detectors Solve This

There are two main strategies:

---

**Strategy 1: Anchors / Dense Predictions (YOLO, SSD)**

Divide the image into a fixed grid. At **every grid cell**, predict a fixed number of boxes (anchors). Then **filter** using a confidence threshold.

```
Image divided into 7×7 grid, 3 anchors per cell = 147 candidate boxes always output

┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ ★ │   │   │   │   │  ← ★ = grid cell with high-confidence object
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │ ★ │   │   │
└───┴───┴───┴───┴───┴───┴───┘

Output: always 147 boxes (fixed size)
Post-processing: keep boxes where confidence > threshold → variable final detections
```

- **Fixed output** from network ✅
- **Variable final detections** after thresholding ✅
- Challenge: most of the 147 boxes are background (class imbalance problem)

---

**Strategy 2: Region Proposals (Faster R-CNN)**

A **Region Proposal Network (RPN)** first scans the image and proposes ~300 candidate regions that *might* contain objects. Then a second stage classifies each proposal.

```
Image → Backbone CNN → Feature Map
                           ↓
                     RPN proposes ~300 regions   ← variable candidate count
                           ↓
                     ROI Pooling (makes each fixed size)
                           ↓
                     Classify each region → final detections
```

- RPN learns to say "something is here" without knowing the class yet
- Variable number of proposals, but each is processed independently

---

**Strategy 3: Set Prediction (DETR — modern)**

Predict exactly $N$ "object slots" (e.g., $N=100$). Each slot either contains a detected object or outputs a special "no object" token $\emptyset$.

```
Output: [obj1, obj2, ∅, ∅, obj3, ∅, ∅, ..., ∅]  ← always 100 slots
                         ↑ empty slots mean "no object here"
```

Uses **Hungarian matching** at training time to assign ground truth objects to slots — no ordering assumption needed.

---

### Summary Table

| Approach | How it handles variable objects | Example |
|----------|--------------------------------|---------|
| **Dense anchors** | Fixed grid of candidates, threshold confidence | YOLO, SSD |
| **Region proposals** | RPN generates variable candidates, classify each | Faster R-CNN |
| **Set prediction** | Fixed N slots, empty slots for "no object" | DETR |

---

**Interview-ready one-liner:**
> "Neural networks have fixed-size outputs, but detection requires a variable number of predictions. This is solved three ways: dense anchor grids (YOLO/SSD) output all candidates then filter by confidence, two-stage detectors (Faster R-CNN) use a region proposal network to generate candidates first, and transformer-based detectors (DETR) use fixed object query slots with Hungarian matching — each is a different architectural answer to the same variable-output problem."

---

## Q: RPN Learns to Say "Something Is Here" Without Knowing the Class

### The Two-Stage Idea

Faster R-CNN splits detection into two clean questions:

- **Stage 1 (RPN):** "Is there *any* object here?" → yes/no + rough box
- **Stage 2 (Detection Head):** "What *class* is it?" + refine box

RPN only answers Stage 1. It is **class-agnostic** — it doesn't care if it's a cat, car, or person. It just says: **"something is here, pay attention to this region."**

---

### What RPN Actually Outputs

At every location on the feature map, RPN predicts two things per anchor:

**1. Objectness score** — a single probability:

$$p_{obj} \in [0, 1]$$

- $p_{obj}$ close to 1 → "I'm confident something is here"
- $p_{obj}$ close to 0 → "This is background, ignore it"

**2. Box delta** — how to adjust the anchor to fit the object:

$$[\delta x, \delta y, \delta w, \delta h]$$

That's it. **No class label. No "cat" or "car".**

```
Feature map location (i, j):

  Anchor 1: p_obj=0.92, [Δx, Δy, Δw, Δh] → "something here, here's a rough box"
  Anchor 2: p_obj=0.08                     → "background, skip"
  Anchor 3: p_obj=0.87, [Δx, Δy, Δw, Δh] → "something here too"
```

---

### How RPN is Trained — What it Learns

The RPN is trained with a **binary label** for each anchor:

| Label | Condition |
|-------|-----------|
| **Positive (object=1)** | Anchor IoU with any GT box ≥ 0.7 |
| **Negative (background=0)** | Anchor IoU with all GT boxes < 0.3 |
| **Ignored** | 0.3 ≤ IoU < 0.7 — ambiguous, not used |

The loss is:

$$L_{RPN} = L_{cls}(p_{obj}, \text{label}) + \lambda \cdot L_{reg}(\delta, \delta^*)$$

- $L_{cls}$ = binary cross-entropy: "object or not?"
- $L_{reg}$ = smooth L1 loss: "how far is the box from GT?"

**Key insight:** The GT label for RPN is just 0 or 1 — derived purely from IoU with ground truth boxes, regardless of what class those boxes are. A GT box for "cat" and a GT box for "car" both produce label=1 for nearby anchors. The RPN never sees the class name.

---

### Visualizing What RPN Sees

```
Ground truth:  [ cat at (100,50,200,150) ]  [ car at (300,80,500,200) ]

RPN only sees:
  ┌──────────────────────────────────────────┐
  │                                          │
  │    ┌──────────┐          ┌──────────┐    │
  │    │ object=1 │          │ object=1 │    │
  │    │ (rough)  │          │ (rough)  │    │
  │    └──────────┘          └──────────┘    │
  │   background=0  background=0             │
  └──────────────────────────────────────────┘

RPN does NOT know: "left one is cat, right one is car"
RPN only knows:    "left region: something, right region: something"
```

---

### Why This Design is Smart

**1. Shared feature computation**

The backbone CNN runs **once** on the whole image. Both RPN and the detection head share these features. No redundant computation.

**2. Focused classification**

By the time Stage 2 runs, it only classifies ~300 high-confidence regions — not millions of raw pixels. This is far easier than classifying everything at once.

**3. Generalizes across classes**

RPN learns what "an object" looks like in general — edges, blobs, distinct regions — not what a specific class looks like. This makes it robust. A network trained on COCO can have its RPN reused for new categories.

**4. Decoupling reduces complexity**

If you tried to do "where is it?" and "what is it?" simultaneously at every pixel, the task is much harder. Separating them lets each stage specialize.

---

### Analogy

Think of RPN as a **security guard at a museum**:

> "I don't know art — I can't tell you if that's a Monet or a Picasso. But I can tell you: *something is hanging on that wall*. Go look at it closely."

Then the **art expert (Stage 2)** goes to exactly those flagged spots and says: "That's a Monet."

The guard doesn't need art knowledge — just the ability to spot that *something is there*.

---

**Interview-ready one-liner:**
> "The RPN is a lightweight binary classifier that slides over the feature map and predicts objectness scores and rough box offsets for each anchor — it has no class vocabulary, just 'object vs background.' This class-agnostic design lets it generalize across all categories, share backbone computation with the detection head, and reduce the classification problem from millions of locations to ~300 high-confidence proposals."

---

## Q: DETR — Set Prediction with N Object Slots

### The Core Idea in Plain English

DETR (Detection Transformer, Facebook 2020) treats object detection as a **set prediction problem**. Instead of anchors or proposals, it says:

> "I will always output exactly 100 candidate detections. Some will be real objects. The rest will say 'nothing here'."

The model learns **which slots to fill** and **which to leave empty**.

---

### What is an "Object Slot"?

Think of DETR's output as a **fixed table with 100 rows**. Each row is one slot:

```
Slot  │  Class predicted     │  Bounding box
──────┼──────────────────────┼───────────────────
  1   │  "person"            │  (0.2, 0.1, 0.1, 0.4)
  2   │  "car"               │  (0.6, 0.5, 0.3, 0.2)
  3   │  ∅ (no object)       │  (ignored)
  4   │  ∅ (no object)       │  (ignored)
  5   │  "dog"               │  (0.4, 0.3, 0.15, 0.2)
  ...
 100  │  ∅ (no object)       │  (ignored)
```

- Slots 1, 2, 5 → real detections
- All other slots → output $\emptyset$ (no object token)
- The **number of real detections varies** per image — but the **output size is always fixed at 100**

---

### What Makes DETR Different — No Anchors, No NMS

| Traditional (YOLO/Faster R-CNN) | DETR |
|--------------------------------|------|
| Thousands of anchors | 100 learned object queries |
| Confidence threshold filtering | Slots predict ∅ directly |
| NMS to remove duplicates | Hungarian matching — each object assigned to exactly one slot |
| Anchors are fixed geometric priors | Object queries are **learned embeddings** |

DETR removes two hand-crafted components entirely: **anchors** and **NMS**.

---

### How Object Queries Work

The 100 slots are driven by **100 learned vectors** called **object queries** $Q \in \mathbb{R}^{100 \times d}$.

Think of each query as a question the model asks:
> "Query 1: Is there something in the top-left area?"
> "Query 47: Is there something large and centered?"

These queries are **not fixed by position** like anchors — they are **learned during training** and can specialize to detect objects of different sizes, positions, or types.

```
Object Queries (learned):
  q_1  →  "specializes in small objects top-left"
  q_2  →  "specializes in large objects center"
  ...
  q_100 → "specializes in objects far right"

(These specializations emerge from training — you don't set them manually)
```

---

### The Full DETR Pipeline

```
Image
  ↓
CNN Backbone (e.g., ResNet)
  ↓
Feature Map  [H/32 × W/32 × 2048]
  ↓
Flatten + Positional Encoding
  ↓
Transformer Encoder  ← image features attend to each other
  ↓
Transformer Decoder  ← 100 object queries attend to encoded image
  ↓
100 output vectors (one per query)
  ↓
FFN (Feed Forward Network) per slot
  ↓
100 predictions: (class or ∅, box coordinates)
```

The **Transformer Decoder** is where the magic happens:
- Each object query "looks at" the entire encoded image via cross-attention
- Queries interact with each other via self-attention → they avoid predicting the same object twice

---

### How Training Works — Hungarian Matching

At training time, you have:
- **100 predicted slots** (some real, some ∅)
- **K ground truth objects** (e.g., K=3 for an image with 3 objects)

You need to decide: **which slot is responsible for which GT object?**

DETR uses the **Hungarian algorithm** — an optimal one-to-one assignment:

```
GT objects:   [cat_GT,  car_GT,  dog_GT]
              ↕ optimal matching (minimize total cost)
Predictions:  [slot_5,  slot_12, slot_1]   ← these 3 slots are matched
              All other 97 slots → assigned label ∅
```

**Matching cost** = classification cost + box regression cost (L1 + GIoU)

The key constraint: **each GT object is matched to exactly one slot** — no duplicates. This is why DETR doesn't need NMS.

---

### What the ∅ Token Means

The $\emptyset$ class is just an extra class index (class 0, or the last class). The model is trained to predict $\emptyset$ for all unmatched slots.

At inference:
- Slots predicting $\emptyset$ → **discarded**
- Slots predicting a real class with high confidence → **kept as detections**

```
Softmax output per slot:

Slot 3:  [person: 0.02, car: 0.01, ..., ∅: 0.97]  → discard (∅ wins)
Slot 7:  [person: 0.91, car: 0.03, ..., ∅: 0.06]  → keep as "person"
```

---

### Why N=100? What if an Image Has >100 Objects?

N=100 is a design choice — large enough for most real-world images (COCO average is ~7 objects per image). If an image genuinely has >100 objects (e.g., dense crowd), DETR will miss some. This is a known limitation.

More recent models (DAB-DETR, DN-DETR) improve this by making queries dynamic and content-aware.

---

### DETR's Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|-----------|
| No anchors, no NMS — fully end-to-end | Slow to train (500 epochs vs 12 for Faster R-CNN) |
| Global context via attention | Poor on small objects (attention is coarse) |
| Clean, elegant architecture | Fixed N limits max detections |
| Naturally handles variable object count | Hungarian matching is expensive |

---

**Interview-ready one-liner:**
> "DETR reformulates detection as set prediction: 100 learned object queries attend to the encoded image via a Transformer decoder, each outputting either a (class, box) prediction or a ∅ 'no object' token. At training, Hungarian matching assigns each ground truth object to exactly one query slot with minimum cost, eliminating the need for anchors and NMS — making DETR the first fully end-to-end object detector."

---

## Q: DETR Removes Anchors and NMS — Why They Existed and Why DETR Doesn't Need Them

---

## Part 1: Anchors — What They Are and Why Traditional Detectors Need Them

### The Problem Anchors Solve

A detector must predict boxes at **many locations and scales**. But raw regression (predicting absolute pixel coordinates) is hard — the numbers are large and unstable.

**Anchors are pre-defined reference boxes** placed at every grid location. Instead of predicting the box from scratch, the model predicts **small corrections** relative to an anchor.

```
Anchor at grid cell (3,4):  [x=96, y=128, w=64, h=64]  ← fixed, pre-defined

Model predicts delta:        [Δx=+5, Δy=-3, Δw=+12, Δh=+20]

Final box = anchor + delta:  [x=101, y=125, w=76, h=84]  ← easy regression task
```

### Why Multiple Anchors Per Location?

Objects come in different shapes. So each grid cell gets **multiple anchors** with different aspect ratios and scales:

```
Each grid cell gets 9 anchors (3 scales × 3 ratios):

  ┌─────────────────┐   ┌─────────┐   ┌───┐
  │  wide anchor    │   │ square  │   │ ↕ │  tall
  └─────────────────┘   └─────────┘   └───┘

  (large)               (medium)      (small)
```

For a 13×13 grid with 9 anchors → **1521 anchor boxes per image.**

### The Problems with Anchors

1. **They are hand-crafted** — you must manually define anchor sizes/ratios using statistics from your dataset (e.g., k-means on GT box dimensions)
2. **Domain-specific** — anchors tuned for COCO work poorly for medical imaging or satellite imagery
3. **Hyperparameter burden** — number of anchors, scales, ratios all need tuning
4. **Imbalance** — 1521 anchors, but only ~5 match GT objects → massive class imbalance

---

## Part 2: NMS — What It Is and Why Traditional Detectors Need It

### The Problem NMS Solves

Because anchors are dense, **multiple nearby anchors all detect the same object**:

```
One cat → triggers 5 overlapping predictions:

  ┌──────────┐
  │┌────────┐│
  ││┌──────┐││
  │││ cat  │││  ← 5 different boxes, all saying "cat" with high confidence
  ││└──────┘││
  │└────────┘│
  └──────────┘
  
Without NMS: model outputs 5 boxes for 1 cat ← wrong
```

### How NMS Works

NMS (Non-Maximum Suppression) is a post-processing step:

1. Sort all predictions by confidence score (highest first)
2. Keep the highest-confidence box
3. **Suppress** (delete) all other boxes with IoU > threshold (e.g., 0.5) with the kept box
4. Repeat for remaining boxes

```
Predictions (sorted):         After NMS:
  box_A: conf=0.95  ← KEEP      box_A: 0.95  ✅
  box_B: conf=0.91  IoU(A,B)=0.8 → SUPPRESS ❌
  box_C: conf=0.88  IoU(A,C)=0.7 → SUPPRESS ❌
  box_D: conf=0.72  IoU(A,D)=0.3 → KEEP (different object) ✅
```

### The Problems with NMS

1. **Not learned** — it's a fixed rule, not trained with the model
2. **IoU threshold is a fragile hyperparameter** — too high → duplicates remain; too low → real objects suppressed
3. **Fails in dense scenes** — two people standing close together may have IoU > 0.5 → one gets suppressed
4. **Not end-to-end** — the network loss doesn't account for NMS behavior during training

---

## Part 3: Why DETR Doesn't Need Either

### No Anchors — Because Object Queries Are Learned

DETR's 100 object queries are **learned embeddings**, not fixed geometric boxes:

```
Traditional:  Anchors are FIXED boxes (defined before training)
              → model learns to correct them

DETR:         Object queries are LEARNED vectors (trained end-to-end)
              → model learns what to look for directly
```

There is no "reference box" to define. Each query learns through training to attend to image regions where objects typically appear. No human decision about scales or ratios.

### No NMS — Because Hungarian Matching Enforces Uniqueness

During training, Hungarian matching ensures **each ground truth object is assigned to exactly one query slot**:

```
Hungarian matching constraint:
  GT object 1  →  slot 7   (one-to-one)
  GT object 2  →  slot_23  (one-to-one)
  GT object 3  →  slot_41  (one-to-one)
  All others   →  ∅

Rule: NO two slots can be assigned the same GT object
```

Because the model is **trained** to never produce duplicates, there are no duplicates at inference — so NMS is unnecessary.

The Transformer's self-attention between queries also helps: queries "talk to each other" during decoding and naturally avoid attending to the same object twice.

---

### Side-by-Side Comparison

```
Traditional Pipeline:              DETR Pipeline:

Image                              Image
  ↓                                  ↓
Backbone                           Backbone
  ↓                                  ↓
Dense anchors (hand-crafted)       Transformer Encoder
  ↓                                  ↓
Predict per anchor                 Transformer Decoder
  ↓                                 (100 learned queries)
1000s of raw predictions            ↓
  ↓                                100 predictions
NMS (hand-crafted rule)             ↓
  ↓                                Final detections
Final detections                   (no post-processing needed)
```

---

**Interview-ready one-liner:**
> "Traditional detectors need anchors because raw box regression is unstable — anchors provide reference geometry — and need NMS because dense anchors cause duplicate detections. DETR eliminates both: it replaces fixed anchors with learned object queries that specialize through training, and replaces NMS with Hungarian matching which enforces a strict one-to-one assignment between queries and ground truth objects during training — so duplicates never arise."

---

## Q: Each Object Query "Looks at" the Entire Image via Cross-Attention

### First, What is Attention in Plain English?

Attention is a mechanism that lets one thing **selectively focus on** parts of another thing.

> "Given what I'm looking for (query), which parts of the image (keys) are most relevant, and what information do I extract from them (values)?"

---

### Three Players in Cross-Attention: Q, K, V

| Symbol | Name | What it is in DETR |
|--------|------|-------------------|
| **Q** | Query | One of the 100 object queries — "what I'm looking for" |
| **K** | Key | Each spatial position in the encoded image — "what's available" |
| **V** | Value | The actual feature content at each position — "what to extract" |

---

### Step-by-Step: How One Query Attends to the Image

**Step 1: The encoded image is a grid of feature vectors**

After the CNN backbone + Transformer encoder, the image becomes a sequence of feature vectors — one per spatial location:

```
Encoded image (flattened feature map):

Position:  p1    p2    p3    p4  ...  pN
           ↓     ↓     ↓     ↓         ↓
Feature:  [f1]  [f2]  [f3]  [f4] ... [fN]   ← each is a vector of dim d

(For a 20×20 feature map: N = 400 positions)
```

Each $f_i$ encodes: "what does the image look like at position $i$?"

---

**Step 2: The query computes a similarity score with every position**

Object query $q$ (a learned vector of dim $d$) is compared to every key $K_i$:

$$\text{score}_i = \frac{q \cdot K_i}{\sqrt{d}}$$

This is a **dot product** — high score = query and key are similar = "this position is relevant to what I'm looking for."

```
Query q (looking for a person):

  score(p1) = 0.1   ← background, not relevant
  score(p2) = 0.8   ← this position has person-like features!
  score(p3) = 0.05  ← sky, not relevant
  score(p4) = 0.75  ← also person-like
  ...
```

---

**Step 3: Softmax converts scores to attention weights**

$$\alpha_i = \text{softmax}(\text{score}_i) = \frac{e^{\text{score}_i}}{\sum_j e^{\text{score}_j}}$$

Now all $\alpha_i$ sum to 1 — they form a **probability distribution over image positions**:

```
Attention weights (visualized on image):

  ┌──────────────────────────────┐
  │ 0.01  0.02  0.01  0.02  0.01 │
  │ 0.02  0.30  0.25  0.05  0.01 │  ← high attention on person region
  │ 0.01  0.15  0.10  0.02  0.01 │
  │ 0.01  0.01  0.01  0.01  0.01 │
  └──────────────────────────────┘
        ↑↑ query focuses here
```

---

**Step 4: Weighted sum of values = the output**

$$\text{output} = \sum_i \alpha_i \cdot V_i$$

The query collects information **proportional to relevance** from every position:

```
output = 0.30 × V(person_region_1)
       + 0.25 × V(person_region_2)
       + 0.15 × V(person_region_3)
       + 0.01 × V(background) + ...
```

Result: a single vector that **summarizes what the query found in the image**.

---

### Why "Entire Image" — Not Just a Local Region

This is the key difference from CNNs. A convolutional filter only sees a **local patch** (e.g., 3×3). Cross-attention computes similarity with **every single position simultaneously**:

```
CNN (local):              Cross-Attention (global):

  ┌───┐                   ┌──────────────────────┐
  │░░░│ ← sees only       │ query attends to ALL │
  └───┘   this patch      │ positions at once    │
                          └──────────────────────┘

For detecting a giraffe:
  CNN needs many layers to "see" head + body together
  Cross-attention: one query can attend to head AND body simultaneously
```

This global receptive field lets DETR reason about **context** — e.g., a query can look at both a person's head and feet in one step to estimate the full box height.

---

### Full Picture in DETR Decoder

```
100 object queries                 Encoded image (N positions)
  [q1, q2, ..., q100]                [f1, f2, ..., fN]
        ↓                                   ↓
        Q (queries)              K, V (keys and values)
             \                      /
              \    Cross-Attention  /
               ↓                  ↓
        Each query outputs a vector summarizing
        "what it found" in the image
               ↓
        FFN → (class prediction, box coordinates)
```

All 100 queries run in **parallel** — each attending to the full image simultaneously. This is why Transformers are GPU-efficient.

---

### Self-Attention Between Queries (Why No Duplicates)

Before cross-attention, queries also do **self-attention among themselves**:

$$q_i \text{ attends to } q_1, q_2, ..., q_{100}$$

This lets them "coordinate":
> "Query 7 is already covering the person on the left — I (query 23) should focus elsewhere."

This coordination is why DETR naturally avoids duplicate detections — queries implicitly negotiate who covers what.

---

**Interview-ready one-liner:**
> "In DETR's cross-attention, each object query computes a dot-product similarity score with every spatial position in the encoded feature map, converts them to attention weights via softmax, then takes a weighted sum of the value vectors — effectively asking 'what does the image contain that matches what I'm looking for?' across all locations simultaneously. This global receptive field in a single operation is fundamentally different from convolution's local patch processing, allowing queries to reason about full object extents and scene context in one step."

---

## Q: How Are Q, K, V Generated in Cross-Attention?

### The Core Idea: They Are All Just Linear Projections

Q, K, and V are not magic — they are created by multiplying input vectors by **learned weight matrices**:

$$Q = X_q \cdot W^Q$$
$$K = X_k \cdot W^K$$
$$V = X_v \cdot W^V$$

Where:
- $X_q$ = the **query source** (object queries in DETR)
- $X_k$ = the **key source** (encoded image features in DETR)
- $X_v$ = the **value source** (encoded image features in DETR — same tensor as $X_k$, but separate projection)
- $W^Q, W^K, W^V$ = **learned weight matrices** (trained via backprop)

> In DETR's cross-attention, $X_k = X_v$ (both are the encoded image), but they are projected through **different weight matrices** — so K and V are different tensors.

---

### Where Each Comes From in DETR

```
Two separate inputs feed into cross-attention:

  Object queries (100 × d)          Encoded image (N × d)
  ────────────────────────          ──────────────────────
  X_q                               X_k = X_v (same source in DETR)
         ↓                           ↓                  ↓
    × W^Q                       × W^K              × W^V
         ↓                           ↓                  ↓
    Q (100 × d)               K (N × d)          V (N × d)
```

- **Q comes from object queries** — "what I'm searching for"
- **K and V both come from the image** — "what's available in the image"

---

### What Are the Weight Matrices $W^Q, W^K, W^V$?

They are **learned projection matrices** — fully connected layers with shape $[d \times d]$ where $d$ is the model dimension (e.g., 256 in DETR).

**Why project at all? Why not use the raw vectors?**

Because the raw object query vector and raw image feature vector serve **different roles**:

- The query should encode **"what pattern am I looking for?"**
- The key should encode **"what pattern do I contain?"**
- The value should encode **"what information should I pass along?"**

Projecting into separate spaces allows the model to **specialize each role independently**:

```
Same image feature f_i:

  f_i × W^K → k_i  "I am a person's torso — matchable by person queries"
  f_i × W^V → v_i  "Here is my rich feature content (color, texture, shape...)"

Same object query q:

  q × W^Q → query  "I am searching for a tall, upright human shape"
```

The dot product $q \cdot k_i$ measures alignment between **what you're searching for** and **what this position contains** — in a learned space.

---

### Concrete Dimension Example (DETR)

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Object queries $X_q$ | $100 \times 256$ | 100 queries, each dim=256 |
| Encoded image $X_k = X_v$ | $400 \times 256$ | 400 positions (20×20 feature map) |
| $W^Q$ | $256 \times 256$ | Learned query projection |
| $W^K$ | $256 \times 256$ | Learned key projection |
| $W^V$ | $256 \times 256$ | Learned value projection |
| Q | $100 \times 256$ | Projected queries |
| K | $400 \times 256$ | Projected keys (one per image position) |
| V | $400 \times 256$ | Projected values (one per image position) |
| Attention scores | $100 \times 400$ | Each query scores every image position |
| Output | $100 \times 256$ | One output vector per query |

---

### The Full Flow — From Raw Inputs to Output

```
Step 1: Project
  Q = X_q × W^Q        → shape [100 × 256]
  K = X_k × W^K        → shape [400 × 256]
  V = X_v × W^V        → shape [400 × 256]

Step 2: Compute scores
  scores = Q × K^T / √256         → shape [100 × 400]
  (each of 100 queries scores all 400 positions)

Step 3: Softmax
  weights = softmax(scores, dim=-1) → shape [100 × 400]
  (each row sums to 1.0)

Step 4: Weighted sum
  output = weights × V             → shape [100 × 256]
  (each query collects a summary of what it found)
```

---

### Why Keys and Values Come From the Same Source but Different Projections

K and V both come from $X_k = X_v$ (image features), but through **different weight matrices**:

- **K** is used for **matching** — "does this position match what the query wants?"
- **V** is used for **information extraction** — "if matched, what do I actually give?"

```
A position might be:
  Easy to match (distinctive key)  but  carry rich information (rich value)

Example:
  Edge of a car door:
    key  = "I'm a horizontal edge"          → matchable by car queries
    value = "color=red, texture=metallic..."  → rich information to extract
```

If K = V (no separate projection), the model loses this flexibility.

---

### How $W^Q, W^K, W^V$ Are Learned

They are initialized randomly and trained end-to-end via backpropagation:

$$\frac{\partial L}{\partial W^Q}, \quad \frac{\partial L}{\partial W^K}, \quad \frac{\partial L}{\partial W^V}$$

Over training, these matrices learn a **query-key space** where:
- Queries for "person" align strongly with keys at human body positions
- Queries for "car" align strongly with keys at vehicle positions
- Background positions get low attention weights from all queries

---

**Interview-ready one-liner:**
> "Q, K, V are linear projections of their source inputs via learned weight matrices: $Q = X_q W^Q$, $K = X_k W^K$, $V = X_v W^V$. In DETR's cross-attention, $X_k = X_v$ (both are encoded image features) but projected separately so K specializes for matching and V specializes for information extraction. The dot product $QK^T/\sqrt{d}$ measures alignment in a learned space, and the softmax-weighted sum over V extracts information proportional to relevance."

---

## Q: Class Imbalance — Background Vastly Outnumbers Objects

### The Simple Intuition First

Imagine a photo of a street. There's 1 car in it.

```
┌──────────────────────────────────────────────────┐
│  sky  sky  sky  sky  sky  sky  sky  sky  sky     │
│  sky  sky  sky  sky  sky  sky  sky  sky  sky     │
│  road road road ┌──────┐ road road road road     │
│  road road road │ CAR  │ road road road road     │
│  road road road └──────┘ road road road road     │
└──────────────────────────────────────────────────┘

Object anchors:      ~5   (match the car with IoU ≥ 0.5)
Background anchors:  ~8000  (everything else)

Ratio: 1600:1 background to object
```

The model sees **thousands of "nothing here"** examples for every **one "object here"** example.

---

### Why This Happens Numerically

In a typical detector (e.g., SSD with 8732 anchors on a 300×300 image):

| Category | Count | % |
|----------|-------|---|
| Positive anchors (match GT objects) | ~10–50 | < 0.5% |
| Negative anchors (background) | ~8700 | > 99% |

This is the **class imbalance problem**.

---

### Why It's a Problem for Training

**The loss becomes dominated by background:**

$$L_{total} = \sum_{\text{all anchors}} L_i = \underbrace{\sum_{\text{8700 negatives}} L_{bg}}_{\text{HUGE}} + \underbrace{\sum_{\text{50 positives}} L_{obj}}_{\text{tiny}}$$

Even if each background loss is small, **8700 small losses overwhelm 50 meaningful ones**.

Effect on training:
- Model learns to **always predict background** — trivially achieves 99% "accuracy" while detecting nothing
- Gradients from background swamp the gradients from real objects
- Model never properly learns object features

```
Naive training:
  Model learns: "Just predict background everywhere"
  Loss is low.  mAP is 0.  Total failure.
```

---

### Three Ways Detectors Fix This

---

**Fix 1: Hard Negative Mining (used in SSD)**

Don't use ALL negative anchors. Rank negatives by their loss (highest loss = model is most wrong) and keep only the **hardest negatives** — those where the model is confidently wrong about background.

$$\text{ratio: 3 hard negatives per 1 positive}$$

```
8700 negatives → sort by loss → keep top 150 hardest
50 positives

Training ratio: 150 : 50 = 3:1  ← manageable
```

**Why "hard"?** Easy negatives (sky, flat road) add no learning signal — the model already knows they're background. Hard negatives (a patch that looks like a face but isn't) force the model to learn the boundary.

---

**Fix 2: Focal Loss (used in RetinaNet — the landmark solution)**

Proposed by Lin et al. (2017). Instead of fixing the ratio, **automatically down-weight easy examples** in the loss function:

$$\text{Focal Loss} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t$ = model's predicted probability for the correct class
- $(1 - p_t)^\gamma$ = **modulating factor** (γ is typically 2)

**How it works:**

| Example type | $p_t$ | $(1-p_t)^2$ | Effect |
|-------------|-------|-------------|--------|
| Easy negative (clearly background) | 0.99 | $(0.01)^2 = 0.0001$ | Loss scaled down 10000× |
| Hard negative (ambiguous) | 0.5 | $(0.5)^2 = 0.25$ | Loss scaled down 4× |
| Positive object | 0.1 | $(0.9)^2 = 0.81$ | Loss almost unchanged |

```
Standard CE loss:             Focal loss:
  background: ████████████      background: █  (down-weighted)
  object:     ██                object:     ██ (unchanged)

Focal loss forces the model to focus on hard, misclassified examples
```

This is why RetinaNet (one-stage) could match Faster R-CNN (two-stage) — focal loss solved the imbalance that previously made one-stage detectors inferior.

---

**Fix 3: Two-Stage Architecture (Faster R-CNN)**

The RPN first filters ~8000 anchors down to ~300 proposals that likely contain objects. The second stage only sees those 300 — already **roughly balanced** between objects and hard negatives.

```
Stage 1 (RPN):   8000 anchors → filter → 300 proposals
Stage 2 (Head):  300 proposals → classify
                 (much better balance — ~50% positive in proposals)
```

This is one reason two-stage detectors were historically more accurate — they naturally sidestepped the imbalance problem.

---

### Summary Table

| Method | How it fixes imbalance | Used in |
|--------|----------------------|---------|
| **Hard Negative Mining** | Subsample negatives, keep hardest | SSD |
| **Focal Loss** | Down-weight easy negatives in loss | RetinaNet, FCOS |
| **Two-stage filtering** | RPN pre-filters to balanced proposals | Faster R-CNN |
| **DETR (Hungarian matching)** | Only matched slots get object loss, others get ∅ — balanced by design | DETR |

---

**Interview-ready one-liner:**
> "In object detection, anchor-based detectors generate thousands of candidate boxes per image but only a handful match ground truth objects — creating a 1000:1 background-to-object ratio. This makes the loss dominated by easy negatives, causing the model to collapse to predicting background everywhere. Solutions include hard negative mining (SSD), focal loss which multiplicatively down-weights easy examples by $(1-p_t)^\gamma$ (RetinaNet), and two-stage architectures where the RPN pre-filters to a balanced proposal set (Faster R-CNN)."

---

## Q: Scale Variation — Objects Vary in Size

### The Simple Intuition First

The same object — a person — can appear at completely different sizes depending on how far they are from the camera:

```
Far away:          Medium distance:     Close up:

   .               ┌──┐               ┌──────────┐
   . ← 10px tall   │  │ ← 80px tall   │          │
   .               │  │               │          │
                   └──┘               │          │
                                      └──────────┘
                                        ← 300px tall

Same object. Same class "person". Completely different scale.
```

A detector must find **all three** — without being told what scale to expect.

---

### Why Scale Variation is Hard

**Problem 1: Fixed receptive field**

A CNN filter has a fixed size (e.g., 3×3). A 3×3 filter:
- Captures enough context for a tiny 10px object
- Captures almost nothing about a 300px object (it only sees 1% of it)

```
3×3 filter on a 300px object:

┌──────────────────────────────┐
│                              │
│   ┌─┐                        │
│   │f│ ← filter sees THIS     │  ← misses the whole object
│   └─┘   tiny region          │
│                              │
└──────────────────────────────┘
```

**Problem 2: Anchor size mismatch**

Anchors are pre-defined at fixed sizes. If a real object is between anchor sizes, no anchor matches it well (IoU < 0.5 → treated as background → missed).

```
Anchors defined at: 32px, 64px, 128px, 256px, 512px

Real object: 45px tall
  IoU with 32px anchor: 0.45 → below threshold → MISSED
  IoU with 64px anchor: 0.40 → below threshold → MISSED
```

**Problem 3: Features at wrong resolution**

Deep layers of a CNN have small spatial resolution (e.g., 7×7 for a 224×224 input). Small objects that were already tiny get **completely lost** in deep feature maps:

```
Input: 224×224  →  VGG16 final layer: 7×7
Scale factor: 32×

A 20px object in the input → 0.6px in the final feature map
→ disappears entirely
```

---

### How Modern Detectors Solve Scale Variation

---

**Solution 1: Image Pyramid (traditional, slow)**

Run the detector at multiple rescaled versions of the same image:

```
Original image (800px)
  ↓ resize
Image at 400px  → detect
Image at 800px  → detect
Image at 1600px → detect
  ↓
Merge all detections
```

- ✅ Simple and effective
- ❌ 3× the computation — too slow for real-time

---

**Solution 2: Feature Pyramid Network — FPN (standard modern approach)**

Instead of resizing the image, **reuse features at multiple depths** of the backbone CNN and combine them:

```
CNN Backbone:

Input (800×800)
  ↓
C2: 200×200  (shallow, fine details, low semantics)
  ↓
C3: 100×100
  ↓
C4:  50×50
  ↓
C5:  25×25   (deep, high semantics, coarse spatial)

FPN adds top-down pathway:

C5 (25×25)  ──upsample──→  P5: 25×25   ← detects large objects
     +C4 lateral connection
C4 (50×50)  ──upsample──→  P4: 50×50   ← detects medium objects
     +C3 lateral connection
C3 (100×100)──upsample──→  P3: 100×100 ← detects small objects
```

Each pyramid level $P_i$ has:
- **High resolution** from shallow layers → can locate small objects
- **Rich semantics** from deep layers → can classify them

```
Small person (20px)  → detected at P3 (fine resolution)
Medium car (100px)   → detected at P4 (medium resolution)
Large truck (400px)  → detected at P5 (coarse resolution)
```

FPN is now the **standard backbone** in Faster R-CNN, RetinaNet, YOLO v3+, and most modern detectors.

---

**Solution 3: Anchor-Free Multi-Scale (FCOS)**

Instead of anchors, predict objects **at the feature map level that best matches the object size**:

$$l_i^* = \text{assign object to level } i \text{ if } m_{i-1} < \max(w, h) \leq m_i$$

Where $m_i$ are scale boundaries (e.g., 64, 128, 256, 512 pixels).

- Objects 0–64px → assigned to P3
- Objects 64–128px → assigned to P4
- Objects 128–256px → assigned to P5

No anchors, no IoU matching — just assign based on object size.

---

**Solution 4: Deformable Convolutions**

Standard convolutions sample from a fixed grid. Deformable convolutions learn **offsets** that adapt the sampling grid to object shape and scale:

```
Standard conv (fixed):      Deformable conv (adaptive):

  ┌─┬─┬─┐                     *   *
  ├─┼─┼─┤       vs           *  *  *      ← samples adapt to object
  └─┴─┴─┘                        *
  (rigid 3×3)                (flexible pattern)
```

Used in Deformable DETR and DCN-based detectors to handle scale and shape variation better.

---

### Why FPN is the Key Innovation to Remember

Before FPN (2017), detectors had to choose: detect at one scale or run slowly at multiple image scales. FPN gave **multi-scale detection for free** by reusing the features the backbone already computes — adding only a lightweight top-down pathway.

```
Cost of FPN ≈ 10% extra computation
Benefit: detects objects across 100× size range
```

This is why mAP on small objects jumped dramatically after FPN was introduced.

---

### Summary Table

| Problem | Solution | Key idea |
|---------|----------|----------|
| Fixed receptive field | FPN | Multi-scale feature maps |
| Anchor size mismatch | Multi-scale anchors per level | Anchors matched to feature level size |
| Small objects disappear | FPN P3 (high res + semantics) | Lateral connections from shallow layers |
| Slow image pyramid | FPN | Reuse backbone features, no re-running |
| Shape variation too | Deformable convolutions | Adaptive sampling grid |

---

**Interview-ready one-liner:**
> "Scale variation means the same object class can span 10px to 500px in the same image. The core solution is Feature Pyramid Networks (FPN), which adds a top-down pathway over backbone feature maps — combining high-resolution shallow features with semantically rich deep features at each level, so small objects are detected at fine-resolution levels and large objects at coarse levels — achieving multi-scale detection at negligible extra cost compared to image pyramids."

---

## Q: Deformable Convolutions — Explained from the Ground Up

### First: How Standard Convolution Samples

A standard 3×3 convolution always samples from a **rigid, fixed grid** of 9 points centered at the current location:

```
Standard 3×3 sampling grid (always the same shape):

  (-1,-1) (0,-1) (+1,-1)
  (-1, 0) (0, 0) (+1, 0)
  (-1,+1) (0,+1) (+1,+1)

Visualized on the feature map:

  ┌───┬───┬───┐
  │ * │ * │ * │
  ├───┼───┼───┤
  │ * │ * │ * │   ← always this exact rectangle
  ├───┼───┼───┤
  │ * │ * │ * │
  └───┴───┴───┘
```

No matter what's in the image — a tiny bird, a huge truck, a tilted face — the filter always looks at the **same rigid rectangle**. It has no ability to adapt.

---

### The Problem This Creates

Objects in real images are:
- **Different sizes** — a 10px bird vs a 400px bus
- **Different shapes** — a horizontal car vs a vertical person
- **Rotated or deformed** — a person bending, a car at an angle

A rigid 3×3 grid treats all of these the same way:

```
Trying to capture a horizontal car with a square grid:

┌───┬───┬───┐
│sky│sky│sky│
├───┼───┼───┤     ← the 3×3 box misses most of the car
│car│car│car│
└───┴───┴───┘

The filter "wants" to look left and right more, but can't
```

---

### The Deformable Convolution Idea

Add a **learnable offset** $(\Delta x_i, \Delta y_i)$ to each of the 9 sampling points. The filter can now sample from **any 9 points in a flexible pattern**:

$$\text{Standard:} \quad y(p) = \sum_{k=1}^{9} w_k \cdot x(p + p_k)$$

$$\text{Deformable:} \quad y(p) = \sum_{k=1}^{9} w_k \cdot x(p + p_k + \Delta p_k)$$

Where:
- $p$ = current center position
- $p_k$ = the fixed grid offset (e.g., $(-1,-1), (0,-1), ...$)
- $\Delta p_k$ = **learned additional offset** — can be any real number (fractional positions use bilinear interpolation)

---

### Where Do the Offsets Come From?

The offsets are predicted by a **separate small convolutional layer** that runs on the same feature map:

```
Feature map
     ↓
  ┌──────────────────────────┐
  │  Offset prediction conv  │  ← lightweight conv layer (same input)
  └──────────────────────────┘
     ↓
  18 offset values per location   (9 points × 2 directions: Δx, Δy)
     ↓
  These offsets deform the sampling grid
     ↓
  ┌──────────────────────────┐
  │  Main deformable conv    │  ← samples from deformed positions
  └──────────────────────────┘
     ↓
  Output feature
```

Both the offset layer and the main conv layer are **trained end-to-end** via backpropagation. The model learns which offsets produce the best features for detection.

---

### Visualizing What Happens

```
Standard conv (fixed grid):         Deformable conv (learned offsets):

  * * *                                *   *
  * * *    ← rigid square             * * *    ← adapts to object shape
  * * *                                  *

For a horizontal car:               For a horizontal car:
  samples miss left/right              offsets push sampling points
  extent of car                        left and right to cover full car

  ┌───┐                               *         *
  │*  │                             *   *   *   *   ← stretches horizontally
  │*  │                               *         *
  └───┘                             (adapts to car's aspect ratio)
```

---

### Concrete Example: What Offsets Learn for Different Objects

After training, the offset predictor learns to produce patterns like:

```
For a large horizontal object (car):
  Offsets stretch the grid wide:
    * . . . * . . . *   ← 3×3 becomes effectively 9×3

For a small square object (face):
  Offsets keep the grid compact:
    * * *
    * * *   ← stays near 3×3
    * * *

For a tilted object (person leaning):
  Offsets rotate the grid:
      *
    *   *
  *       *   ← tilted sampling pattern
```

The network **automatically learns** these patterns from data — no manual rules needed.

---

### How Fractional Offsets Work — Bilinear Interpolation

Offsets like $\Delta p = (1.7, -0.4)$ don't land on exact pixel positions. The value is computed by **bilinear interpolation** between the 4 nearest integer positions:

$$x(p + \Delta p) = \sum_{q} G(q, p + \Delta p) \cdot x(q)$$

Where $G(q, p') = \max(0, 1 - |q_x - p'_x|) \cdot \max(0, 1 - |q_y - p'_y|)$

This makes the operation **differentiable** — gradients flow back through the offsets during training.

---

### Why This Helps Scale Variation Specifically

| Scenario | Standard conv | Deformable conv |
|----------|--------------|-----------------|
| Large object | Sees tiny patch of it | Offsets spread out to cover more |
| Small object | Overshoots, includes background | Offsets contract to focus tightly |
| Elongated object | Square grid misses extent | Offsets stretch along long axis |
| Rotated object | Grid misaligned | Offsets rotate to align with object |

---

### Deformable Conv v1 vs v2

| Version | What it adds |
|---------|-------------|
| **DCNv1** (2017) | Learnable spatial offsets $\Delta p_k$ |
| **DCNv2** (2019) | Offsets + **modulation weights** $\Delta m_k$ per sample point — can also suppress irrelevant points |

DCNv2 formula:
$$y(p) = \sum_{k=1}^{9} w_k \cdot \Delta m_k \cdot x(p + p_k + \Delta p_k)$$

$\Delta m_k \in [0,1]$ — if a sampled point lands on background/noise, the network can zero it out.

---

**Interview-ready one-liner:**
> "Standard convolutions sample from a fixed 3×3 grid regardless of object shape or scale. Deformable convolutions add a learned offset $\Delta p_k$ to each of the 9 sampling locations, predicted by a lightweight parallel conv layer — so the sampling pattern adapts to object geometry: stretching wide for horizontal cars, contracting for small faces, rotating for tilted objects. The offsets are trained end-to-end via backprop through bilinear interpolation, giving the model geometric adaptability without any hand-crafted rules."

---

*End of notes — continued in next session.*
