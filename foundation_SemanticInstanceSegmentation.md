# Semantic & Instance Segmentation — Deep Foundational Notes
*Interview prep for Senior Computer Vision Researcher roles*

---

## Q: Semantic Segmentation — Class Label Per Pixel, Doesn't Distinguish Instances

### Start from the Simplest Task: Image Classification

```
Image Classification:
┌────────────────────┐
│                    │   → ONE label for the WHOLE image
│   🐱   🐱   🐱    │   → "cat"
│                    │
└────────────────────┘
Output: "cat"  (1 number)
```

---

### Object Detection: One Box Per Object

```
Object Detection:
┌────────────────────┐
│  ┌────┐  ┌────┐   │   → boxes with labels
│  │cat │  │cat │   │   → (cat, box1), (cat, box2)
│  └────┘  └────┘   │
└────────────────────┘
Output: [(cat, x1,y1,w1,h1), (cat, x2,y2,w2,h2)]
```

---

### Semantic Segmentation: ONE Label Per PIXEL

```
Semantic Segmentation:
┌────────────────────┐        ┌────────────────────┐
│                    │        │ 0  0  0  0  0  0   │
│   🐱   🐱   🐱    │  →     │ 0  1  1  1  1  0   │  1 = "cat"
│   (grass below)    │        │ 0  1  1  1  1  0   │  2 = "grass"
│                    │        │ 0  2  2  2  2  0   │  0 = "background"
└────────────────────┘        └────────────────────┘
Input image                   Output: label map (same H×W as image)
                              Every single pixel gets exactly ONE class label
```

The output is a **label map** — a 2D grid the same size as the image where every pixel has a class ID.

---

### What "Semantic-Level" Means

"Semantic" means the model understands **what class of thing** a pixel belongs to — not **which specific individual** it is.

```
Image with two cats:

┌───────────────────────────────┐
│     cat A      │    cat B     │
│  (sitting)     │  (walking)   │
└───────────────────────────────┘

Semantic segmentation output:

┌───────────────────────────────┐
│  cat  cat  cat │ cat  cat cat │
│  cat  cat  cat │ cat  cat cat │
└───────────────────────────────┘
        ↑
ALL pixels labeled "cat" — no distinction between cat A and cat B
```

The model knows **"this is cat pixels"** — but it doesn't know **"this is cat #1, that is cat #2."**

---

### Why "Doesn't Distinguish Instances" is a Fundamental Limitation

**Instance** = one specific individual object.

```
Two cats touching each other:

┌──────────────────────┐
│ cat A ┃ cat B        │   ← they are touching
│       ┃              │
└──────────────────────┘

Semantic segmentation sees:

┌──────────────────────┐
│  cat   cat   cat     │   ← one big blob of "cat"
│  cat   cat   cat     │   ← boundary between A and B is LOST
└──────────────────────┘

The model merges them — you cannot tell where A ends and B begins
```

This is the core limitation. Semantic segmentation answers:
> **"What is at each pixel?"**

But NOT:
> **"Which specific object does each pixel belong to?"**

---

### Comparing All Three Tasks Side by Side

```
Same image: 2 cats on grass

┌──────────────────────────────────────────────────────────┐
│         │ Object Detection  │ Semantic Seg  │ Instance Seg│
├──────────┼───────────────────┼───────────────┼────────────┤
│ Output   │ 2 boxes + labels  │ pixel labels  │ pixel labels│
│          │ (cat,cat)         │ (cat/grass)   │ + IDs      │
├──────────┼───────────────────┼───────────────┼────────────┤
│ Cat A vs │ ✅ separate boxes │ ❌ merged     │ ✅ separate │
│ Cat B    │                   │  "cat" blob   │  masks     │
├──────────┼───────────────────┼───────────────┼────────────┤
│ Pixel    │ ❌ only boxes,    │ ✅ every pixel│ ✅ every    │
│ precise  │   not pixel-level │   labeled     │ pixel      │
└──────────┴───────────────────┴───────────────┴────────────┘
```

---

### What the Output Tensor Looks Like

For an image of size $H \times W$ with $C$ classes:

**Semantic segmentation raw output** (before argmax):

$$\text{logits} \in \mathbb{R}^{H \times W \times C}$$

At each pixel $(i, j)$, the model outputs $C$ scores — one per class.

**Final prediction** (after argmax):

$$\text{label}(i, j) = \arg\max_c\ \text{logits}(i, j, c)$$

```
Pixel (100, 200):
  scores: [background=0.1, cat=0.85, grass=0.05]
  prediction: "cat"  ← class with highest score

Pixel (300, 200):
  scores: [background=0.05, cat=0.1, grass=0.85]
  prediction: "grass"
```

---

### Real-World Applications Where "No Instance Distinction" is Acceptable

| Application | Why semantic is enough |
|-------------|----------------------|
| **Road segmentation** (autonomous driving) | You need to know "this is road" — you don't need "road #1 vs road #2" |
| **Sky removal** (photo editing) | "These are sky pixels" is sufficient |
| **Organ segmentation** (medical imaging) | "These pixels are liver" — there's only one liver |
| **Terrain mapping** (satellite) | "Forest, water, urban" — classes matter, not individual trees |

But for **counting people**, **tracking individuals**, or **separating touching objects** — you need instance segmentation.

---

**Interview-ready one-liner:**
> "Semantic segmentation assigns a single class label to every pixel in the image — the output is a label map of the same spatial dimensions as the input. It operates at the semantic level: it answers 'what class is at each pixel?' but not 'which specific instance?' — so two touching cats would both be labeled 'cat' with no boundary between them. This limitation is why instance segmentation was developed, which additionally assigns a unique instance ID to each individual object's pixels."

---

## Q: Pixel-Level Detail — Need to Preserve Fine Spatial Information

### The Core Problem

In semantic segmentation, the final output must be **the same size as the input image** — every single pixel needs a label. But standard CNN architectures are designed to **progressively shrink** spatial resolution to extract abstract features.

These two goals are in direct conflict:

```
CNN's natural behavior:          What segmentation needs:

Input:   224×224                 Output must be: 224×224
  ↓ conv + pool                    (same size as input)
  64×64
  ↓ conv + pool
  28×28
  ↓ conv + pool
  7×7   ← tiny!                 ← can't predict pixel labels from 7×7
```

---

### Why CNNs Shrink Spatial Resolution

Two operations cause this shrinkage:

**1. Stride in convolutions**

A conv with stride=2 halves the spatial dimensions:

```
Input feature map: 64×64
Conv stride=2 →    32×32   (every 2nd position sampled)
```

**2. Pooling layers (Max/Average pool)**

```
Input: 64×64
Max pool 2×2, stride=2 → 32×32
```

After 5 such stages (typical backbone):

$$224 \div 2^5 = 7$$

A $224 \times 224$ image becomes a $7 \times 7$ feature map — **32× smaller in each dimension**.

---

### Why Small Feature Maps Lose Fine Spatial Detail

```
Original image (224×224):           Feature map (7×7):

┌──────────────────────┐            ┌───────┐
│  person's finger     │            │       │
│  ←── 3px wide ───→   │  →shrink→  │  ·    │  ← finger is now 0.09px
│                      │            │       │      COMPLETELY LOST
└──────────────────────┘            └───────┘

Each cell in 7×7 = 32×32 pixel region in original image
Fine details smaller than 32px → invisible at this resolution
```

**What gets lost:**
- Thin object boundaries (edges of a person vs background)
- Small objects (a traffic sign, a pedestrian far away)
- Fine-grained structures (fingers, hair strands, thin poles)

---

### What "Preserving Fine Spatial Information" Means

To assign labels at pixel precision, the model must remember **where things are** at the original resolution — not just **what** they are.

Two types of information exist at different network depths:

```
Shallow layers (high resolution):     Deep layers (low resolution):
  ← fine spatial detail →               ← rich semantic meaning →

  ┌──┬──┬──┬──┬──┬──┬──┐               ┌───────┐
  │  │  │  │  │  │  │  │               │       │
  ├──┼──┼──┼──┼──┼──┼──┤               │       │
  │  │  │  │  │  │  │  │               └───────┘
  └──┴──┴──┴──┴──┴──┴──┘
  "I see an edge at (x=102, y=47)"      "This region is a person"
  (precise location, low semantics)     (rich meaning, imprecise location)
```

**The challenge:** you need BOTH simultaneously for pixel-level segmentation:
- Deep layers tell you **"this is a cat"** but can't tell you exactly **which pixels**
- Shallow layers tell you **exactly where** boundaries are but don't know **what** they belong to

---

### How Architectures Solve This — Three Main Approaches

---

**Approach 1: Naive Upsampling (simple but blurry)**

Just upsample the 7×7 feature map back to 224×224 using bilinear interpolation:

```
7×7 feature map
  ↓ bilinear upsample ×32
224×224 prediction

Problem:
┌──────────────────────┐
│  cat  cat  cat  cat  │
│  cat  cat  cat  cat  │   ← blocky, blurry boundaries
│  cat  cat  cat  cat  │   ← no fine edge detail
│  bground bground bg  │
└──────────────────────┘
```

The upsampled map is **spatially coarse** — 32 pixels in the output all get the same label from one 7×7 cell.

---

**Approach 2: Skip Connections — FCN (2015, revolutionary)**

The key insight: **reuse fine-detail feature maps from earlier layers** to guide the upsampling.

```
Encoder (backbone):

Input 224×224
  ↓
conv1: 112×112  ← fine edges, low semantics
  ↓
conv2:  56×56
  ↓
conv3:  28×28
  ↓
conv4:  14×14
  ↓
conv5:   7×7   ← coarse, high semantics

Decoder (FCN with skip connections):

conv5 (7×7)
  ↓ upsample ×2
  14×14 + conv4 (14×14)  ← ADD fine details from conv4
  ↓ upsample ×2
  28×28 + conv3 (28×28)  ← ADD more fine details
  ↓ upsample ×8
  224×224 prediction
```

```
Visual effect:

Without skip:   ──────────────────── blurry boundary
With skip:      ─────────|──────────  sharp boundary
                         ↑ fine edge from shallow layer
```

---

**Approach 3: Encoder-Decoder with Symmetric Architecture (U-Net)**

U-Net (2015, medical imaging) takes skip connections further — the encoder and decoder are **symmetric mirrors**:

```
Encoder (contracts):        Decoder (expands):

224×224 ──────────────────────────────→ 224×224
  ↓                                         ↑
112×112 ────────────────────────────→ 112×112
  ↓                                         ↑
 56×56  ──────────────────────────→  56×56
  ↓                                         ↑
 28×28  ────────────────────────→   28×28
  ↓                                         ↑
 14×14  ──────────────────────→    14×14
  ↓                                         ↑
  7×7   → bottleneck → upsample ────→  7×7

  ↑────────── skip connections ────────────↑
  (fine spatial info passed directly across)
```

Each skip connection **concatenates** the encoder feature map directly to the decoder at the same resolution — preserving exact spatial positions of boundaries, edges, and fine structures.

---

**Approach 4: Dilated/Atrous Convolutions (DeepLab series)**

Instead of shrinking the feature map, use **dilated convolutions** to maintain resolution while expanding the receptive field:

```
Standard 3×3 conv (rate=1):    Dilated 3×3 conv (rate=2):

  * * *                          *   *   *
  * * *   receptive = 3×3        
  * * *                          *   *   *   receptive = 5×5
                                             (same 9 params, larger coverage)
                                 *   *   *
```

Dilation rate $r$ inserts $r-1$ zeros between filter weights:
- rate=1 → standard conv, receptive field = 3×3
- rate=2 → receptive field = 5×5 (no extra parameters)
- rate=4 → receptive field = 9×9

DeepLab uses **Atrous Spatial Pyramid Pooling (ASPP)** — parallel dilated convolutions at multiple rates to capture multi-scale context **without losing resolution**:

```
Feature map (28×28, not 7×7 — resolution preserved by removing last 2 pools)
  ↓ dilated conv rate=6  → 28×28
  ↓ dilated conv rate=12 → 28×28
  ↓ dilated conv rate=18 → 28×28
  ↓ 1×1 conv             → 28×28
  → concat all → 28×28 → upsample ×8 → 224×224
```

---

### Summary: The Trade-off at Every Architecture Level

```
Resolution:   High ←───────────────────────→ Low
              (fine spatial detail)           (rich semantics)

Shallow CNN:  ████████████████               (lots of detail, low meaning)
Deep CNN:                     ████████████   (little detail, rich meaning)

Segmentation needs BOTH simultaneously:
Solution:     Skip connections / U-Net / Dilated convs
```

---

**Interview-ready one-liner:**
> "Preserving fine spatial information is the central architectural challenge in semantic segmentation — standard CNN backbones reduce spatial resolution by 32× through pooling and strided convolutions, which destroys pixel-level boundary precision. Solutions include FCN's skip connections (reusing shallow high-resolution feature maps during upsampling), U-Net's symmetric encoder-decoder with full-resolution skip concatenations, and DeepLab's dilated convolutions which expand receptive field without reducing resolution — each is a different strategy to maintain 'where' information while still capturing 'what' information from deep layers."

---

## Q: Dilated/Atrous Convolutions — Explained from Ground Up

### First: The Problem to Solve

Standard CNNs shrink feature maps to build large receptive fields:

```
To "see" a 65×65 region with 3×3 convs:
  Need 32 stacked conv layers OR
  Need 5 pooling layers (32× downsampling)

After 5 poolings: 224×224 → 7×7
Fine spatial detail is destroyed
```

**The dilemma:**
- **Large receptive field** → needed to understand context (what class is this?)
- **High resolution** → needed to know exact pixel locations (where exactly?)

Standard convolutions force you to choose one. Dilated convolutions give you **both**.

---

### What "Dilation" Actually Means

Dilation inserts **gaps (zeros) between filter weights** before applying the convolution. The filter itself doesn't change — only how it samples the input.

**Standard 3×3 conv (dilation rate = 1):**

```
Filter weights:        Applied to input:
  w1  w2  w3            w1  w2  w3
  w4  w5  w6      →     w4  w5  w6     ← samples 9 consecutive positions
  w7  w8  w9            w7  w8  w9

Receptive field: 3×3
```

**Dilated 3×3 conv (dilation rate = 2):**

```
Filter weights:        Applied to input (gaps inserted):
  w1  w2  w3            w1  .  w2  .  w3
  w4  w5  w6      →     .   .   .  .   .    ← skips every other position
  w7  w8  w9            w4  .  w5  .  w6
                         .   .   .  .   .
                        w7  .  w8  .  w9

Receptive field: 5×5  (covers same area as 5×5 conv)
Number of parameters: still only 9  (same as 3×3)
```

**Dilated 3×3 conv (dilation rate = 4):**

```
Receptive field: 9×9  (covers same area as 9×9 conv)
Number of parameters: still only 9
```

---

### The General Formula

For a 3×3 filter with dilation rate $r$, the effective kernel size:

$$\text{Effective kernel size} = k + (k-1)(r-1)$$

For $k=3$:

| Dilation rate $r$ | Effective kernel size | Receptive field |
|-------------------|----------------------|-----------------|
| 1 | 3 | 3×3 |
| 2 | 5 | 5×5 |
| 3 | 7 | 7×7 |
| 4 | 9 | 9×9 |
| 6 | 13 | 13×13 |
| 12 | 25 | 25×25 |
| 18 | 37 | 37×37 |

**Same 9 parameters. Enormous receptive field. Resolution unchanged.**

---

### Visual: What Gets Sampled

```
Input feature map (simplified):

. . . . . . . . . . . . .
. . . . . . . . . . . . .
. . * . * . * . . . . . .   ← rate=2: samples here
. . . . . . . . . . . . .
. . * . * . * . . . . . .
. . . . . . . . . . . . .
. . * . * . * . . . . . .
. . . . . . . . . . . . .

9 sample points, but spread across a 5×5 area

Compare to rate=1 (standard):
. . . . . . . . .
. . . . . . . . .
. . * * * . . . .   ← samples consecutive positions
. . * * * . . . .
. . * * * . . . .
. . . . . . . . .

9 sample points, only covering 3×3 area
```

---

### Why Resolution is Maintained

Standard approach to increase receptive field:
```
Conv → Pool → Conv → Pool  →  small feature map, large receptive field
224×224 → 112×112 → 56×56 → 28×28  ← SHRINKS
```

Dilated approach:
```
Conv(r=1) → Conv(r=2) → Conv(r=4) → Conv(r=8)  →  same spatial size, large receptive field
224×224   →  224×224  →  224×224  →  224×224   ← STAYS SAME
```

DeepLab achieves this by:
1. Removing the last 2 max-pooling layers from ResNet
2. Replacing subsequent convolutions with dilated convolutions

```
Standard ResNet:              DeepLab modification:

conv4: 28×28                  conv4: 28×28
  ↓ pool                        ↓ (pool REMOVED)
conv5: 14×14                  conv5 with dilation=2: 28×28  ← same resolution!
  ↓ pool                        ↓ (pool REMOVED)
conv6:  7×7                   conv6 with dilation=4: 28×28  ← still same!
```

Final feature map: **28×28 instead of 7×7** — 16× more spatial information preserved.

---

### ASPP — Atrous Spatial Pyramid Pooling

A single dilation rate captures one scale. But objects appear at **multiple scales** in real images. ASPP runs **multiple dilated convolutions in parallel**, each at a different rate:

```
Input feature map (28×28)
         ↓
  ┌──────┬──────┬──────┬──────┐
  ↓      ↓      ↓      ↓      ↓
1×1    r=6    r=12   r=18  Global
conv   dil    dil    dil   Avg
                           Pool
  ↓      ↓      ↓      ↓      ↓
 28×28  28×28  28×28  28×28  1×1
                              ↓ upsample to 28×28
  └──────┴──────┴──────┴──────┘
              ↓ concat
          28×28 × (256×5)
              ↓
          1×1 conv (reduce channels)
              ↓
          28×28 × 256
              ↓
          upsample ×8
              ↓
          224×224 prediction
```

**Why each branch captures different things:**

| Branch | What it sees |
|--------|-------------|
| 1×1 conv | Point-level features, no context |
| r=6 | Small objects, local context (13×13 receptive field) |
| r=12 | Medium objects, wider context (25×25 receptive field) |
| r=18 | Large objects, broad context (37×37 receptive field) |
| Global avg pool | Full image context (scene-level) |

All 5 branches are **concatenated** — the model gets multi-scale context at full 28×28 resolution simultaneously.

---

### Comparing DeepLab Versions

| Version | Key addition |
|---------|-------------|
| **DeepLab v1** (2015) | Dilated convolutions + CRF post-processing |
| **DeepLab v2** (2016) | ASPP (multi-rate parallel dilated convs) |
| **DeepLab v3** (2017) | Improved ASPP + global average pooling branch, removed CRF |
| **DeepLab v3+** (2018) | Added decoder with skip connections (like U-Net) on top of v3 |

---

### The "Gridding Artifact" Problem with High Dilation

One known issue: high dilation rates (e.g., r=16) sample positions that are **too spread out**, missing information between sample points:

```
Rate=8 sampling pattern:

*  .  .  .  .  .  .  *  .  .  .  .  .  .  *
(15 pixels between each sample point — huge gaps)
```

This causes **gridding artifacts** in the output — repeated patterns of missed information. DeepLab v3 mitigates this by using **multiple rates in cascade** rather than one very high rate.

---

**Interview-ready one-liner:**
> "Dilated (atrous) convolutions insert gaps of rate $r$ between filter weights, expanding the effective receptive field from 3×3 to $(2r+1) \times (2r+1)$ while keeping the same 9 parameters and — crucially — without reducing spatial resolution. DeepLab removes the last two pooling layers from ResNet and replaces subsequent convolutions with dilated ones, preserving a 28×28 feature map instead of 7×7. ASPP then runs parallel dilated convolutions at rates 6, 12, 18 simultaneously to capture multi-scale context at full resolution — solving the resolution-vs-receptive-field dilemma that limits standard encoder-decoder architectures."

---

## Q: FCN — Fully Convolutional Network, Explained from Ground Up

### First: What is a Fully Connected (FC) Layer?

A standard classification CNN (like AlexNet/VGG) ends with **fully connected layers**:

```
Input image (224×224×3)
  ↓
Conv layers → feature maps
  ↓
Flatten: 7×7×512 = 25,088 values → one long vector
  ↓
FC layer 1: 25,088 → 4,096 neurons
  ↓
FC layer 2: 4,096 → 4,096 neurons
  ↓
FC layer 3: 4,096 → 1,000 classes (softmax)
  ↓
Output: [dog: 0.9, cat: 0.05, ...]  ← ONE label for whole image
```

**The flatten + FC structure destroys all spatial information** — the 7×7 grid location is lost when everything is collapsed to a 1D vector.

---

### The Two Critical Problems with FC Layers for Segmentation

**Problem 1: Fixed input size**

The FC layer has a **fixed number of input neurons** (e.g., 25,088). This means the network only accepts **exactly 224×224** images. Feed it a 300×400 image → crash. The FC layer is hard-wired to one size.

**Problem 2: No spatial output**

FC layers output a single vector (e.g., 1000 class scores for the whole image). There is no way to get per-pixel predictions from an FC layer — all spatial structure is gone after the flatten.

```
What we need for segmentation:

Input:  224×224 image
Output: 224×224 label map  ← spatial output, same size as input

FC layers give us:
Output: [1000 numbers]  ← not spatial at all
```

---

### The FCN Key Insight: FC Layers ARE Convolutions in Disguise

**This is the core mathematical insight of FCN (Long et al., 2015).**

A fully connected layer applied to a $7 \times 7 \times 512$ feature map:

$$\text{FC}: \underbrace{7 \times 7 \times 512}_{25088} \rightarrow 4096$$

Can be **exactly rewritten** as a $7 \times 7$ convolution with 4096 filters:

$$\text{Conv } 7 \times 7: \underbrace{7 \times 7 \times 512}_{\text{input}} \xrightarrow{\text{4096 filters of size } 7\times7} \underbrace{1 \times 1 \times 4096}_{\text{output}}$$

They compute **identical values** — both multiply the same weights by the same inputs and sum them. The only difference is notation.

```
FC layer (when input is 7×7×512):

  Flatten → [25088 values] → matrix multiply W[4096×25088] → [4096 values]

≡ Conv 7×7 with 4096 filters (same weights, just reshaped):

  [7×7×512] → conv with kernel 7×7×512, 4096 filters → [1×1×4096]

Same computation. Same result. Different framing.
```

---

### What Happens When You Replace FC with Conv on a LARGER Image

This is where it gets powerful. If the input image is larger (e.g., $384 \times 384$ instead of $224 \times 224$):

**With FC layers:** ❌ Crash — fixed weight matrix expects exactly 25088 inputs

**With equivalent Conv 7×7:** ✅ Works — the conv slides over the larger feature map

```
Input: 384×384 (instead of 224×224)

After 5 conv+pool stages: 12×12×512 (instead of 7×7×512)

Apply "Conv 7×7" (the converted FC layer):
  12×12×512 → conv 7×7 → 6×6×4096  ← produces a 6×6 spatial grid!

Each cell in the 6×6 grid = one "classification" for that region of the image
```

The network **naturally produces a spatial output map** when given a larger image. Each position in the output map corresponds to a region of the input image.

---

### The Complete FCN Architecture

```
Standard VGG (classification):          FCN (segmentation):

Input: 224×224×3                        Input: any H×W×3  ✅

Conv1: 224×224×64                       Conv1: H×W×64
Conv2: 224×224×64                       Conv2: H×W×64
Pool:  112×112×64                       Pool:  H/2×W/2×64

Conv3: 112×112×128                      Conv3: H/2×W/2×128
Pool:   56×56×128                       Pool:  H/4×W/4×128

Conv4:  56×56×256                       Conv4: H/4×W/4×256
Pool:   28×28×256                       Pool:  H/8×W/8×256

Conv5:  28×28×512                       Conv5: H/8×W/8×512
Pool:   14×14×512                       Pool:  H/16×W/16×512

Conv6:  14×14×512                       Conv6: H/16×W/16×512
Pool:    7×7×512                        Pool:  H/32×W/32×512

Flatten → 25088                         ← NO FLATTEN
FC: 25088→4096                          Conv 7×7: → H/32×W/32×4096
FC: 4096→4096                           Conv 1×1: → H/32×W/32×4096
FC: 4096→1000                           Conv 1×1: → H/32×W/32×21  (21 PASCAL classes)

Output: [1000 scores]                   Output: H/32×W/32×21  ← spatial map!
(one for whole image)                   (coarse predictions per region)
```

For a 224×224 input: final map is **7×7×21** — 7×7 coarse predictions, one per 32×32 pixel region.

---

### The Upsampling Problem — From Coarse to Dense

The 7×7 spatial map needs to be upsampled back to 224×224 for pixel-level predictions:

**FCN-32s (naive):** upsample directly ×32

```
7×7 → bilinear upsample ×32 → 224×224

Result: very blurry — each 32×32 block gets same label
```

**FCN-16s (1 skip connection):** add pool4 features

```
pool4: 14×14×512  ─────────────────────────────┐
                                                ↓
conv7 (7×7×21) → upsample ×2 → 14×14×21 → add → upsample ×16 → 224×224
```

**FCN-8s (2 skip connections):** add pool3 features too

```
pool3: 28×28×256  ──────────────────────────────────┐
pool4: 14×14×512  ─────────────────────┐            │
                                       ↓            ↓
conv7 → upsample×2 → 14×14 → add → upsample×2 → 28×28 → add → upsample×8 → 224×224
```

```
Quality comparison:

FCN-32s:  ████████████████   blurry, coarse
FCN-16s:  ████████|████████  better boundaries
FCN-8s:   ████|████|████|██  sharp boundaries  ← best
                ↑ skip connections add fine detail
```

---

### Why "Variable-Size Input" is Revolutionary

Before FCN, every segmentation approach needed a fixed-size input because of FC layers. FCN's all-convolutional design means:

```
Train on:  224×224 images
Test on:   any size — 300×400, 1024×768, 2048×1024

The network slides its "classifier" over the full image simultaneously
→ Dense predictions in ONE forward pass
→ No sliding window needed
→ Efficient: shared computation across all positions
```

This is the "dense prediction" aspect — predict for all pixels in one shot, not one patch at a time.

---

### Why "End-to-End Trainable" Matters

Before FCN, segmentation pipelines had multiple **separate** components:
1. Extract hand-crafted features (SIFT, HOG)
2. Run superpixel algorithm
3. Train separate classifiers per region
4. Post-process with CRF

Each step trained independently — errors in step 1 can't be corrected by step 3.

FCN trains **all layers jointly** with a single pixel-wise cross-entropy loss:

$$L = -\frac{1}{H \times W} \sum_{i,j} \log p(y_{ij} | x)$$

Gradients flow from the pixel-level loss all the way back through upsampling, skip connections, and convolutions — **every layer improves together** toward the same goal.

---

### Summary

| Property | Standard CNN | FCN |
|----------|-------------|-----|
| FC layers | Yes — fixed size, no spatial output | Replaced by 1×1 and 7×7 convs |
| Input size | Fixed (e.g., 224×224 only) | Any size ✅ |
| Output | Single class vector | Spatial prediction map ✅ |
| Per-pixel labels | ❌ Impossible | ✅ Natural output |
| End-to-end training | Classification only | Full pixel-level supervision ✅ |

---

**Interview-ready one-liner:**
> "FCN's key insight is that a fully connected layer applied to a $7\times7$ feature map is mathematically identical to a $7\times7$ convolution — same weights, same computation, just reshaped. By replacing all FC layers with their convolutional equivalents, the network loses its fixed-size constraint and naturally produces a spatial output map instead of a single class vector. On larger inputs the convolutional 'classifier' slides over the feature map producing dense predictions in one forward pass. Skip connections from pool3 and pool4 recover fine spatial detail lost during downsampling — making FCN the first end-to-end trainable network for pixel-level dense prediction."

---

## Q: Why Flatten + FC Destroys Spatial Information

### What "Spatial Information" Means

Spatial information = **where something is** in the image.

A feature map preserves location:

```
Feature map (7×7×512):

  Position (0,0) (0,1) (0,2) ... (0,6)
           (1,0) (1,1) (1,2) ... (1,6)
           ...
           (6,0) (6,1) (6,2) ... (6,6)

Each position (i,j) corresponds to a specific region in the original image:
  (0,0) → top-left region of image
  (3,3) → center of image
  (6,6) → bottom-right region of image
```

As long as data stays in the 2D grid format, the network **knows where things are**.

---

### What Flattening Does

Flatten takes the 2D spatial grid and **unrolls it into one long list**:

```
Before flatten (7×7×512 feature map):

  Spatial grid — position matters:
  
  [top-left features]  [top-center features]  [top-right features]
  [mid-left features]  [center features]      [mid-right features]
  [bot-left features]  [bot-center features]  [bot-right features]

After flatten (25,088 values, 1D vector):

  [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, ... f25088]
   ↑                                                   ↑
  was top-left                               was bottom-right
  
  BUT NOW: there is no "top-left" or "bottom-right" — just index 0 and index 25087
```

The 2D structure is gone. All 25,088 values are just **a numbered list with no geometry**.

---

### Concrete Example: A Cat in the Corner vs Center

```
Image A: cat in top-left corner

Feature map:
  [cat features]  [bg]  [bg]
  [bg]            [bg]  [bg]
  [bg]            [bg]  [bg]

Flattened:
  [cat_f1, cat_f2, cat_f3, bg, bg, bg, bg, bg, bg, ...]
   ↑ indices 0,1,2 are "cat"


Image B: cat in center

Feature map:
  [bg]  [bg]          [bg]
  [bg]  [cat features] [bg]
  [bg]  [bg]          [bg]

Flattened:
  [bg, bg, bg, bg, cat_f1, cat_f2, cat_f3, bg, bg, ...]
                    ↑ indices 4,5,6 are "cat"
```

**Both images have a cat — but the cat features appear at completely different indices in the flattened vector.** The FC layer sees them as totally different inputs, even though semantically they're the same thing at different locations.

The FC layer has **no concept of adjacency or position** — index 0 and index 1 are not "neighbors" to it, they're just two separate numbers.

---

### What the FC Layer Actually Computes

An FC layer is just a **matrix multiplication**:

$$\text{output} = W \cdot \text{input} + b$$

Where $W$ has shape $[4096 \times 25088]$.

Each output neuron is a **weighted sum of ALL 25,088 input values**:

```
output_neuron_47 = w_{47,0}×f_0 + w_{47,1}×f_1 + ... + w_{47,25087}×f_{25087}
```

Every single input value contributes to every output neuron — **regardless of where in the image it came from**. The weight $w_{47, 0}$ (for the top-left feature) and $w_{47, 25087}$ (for the bottom-right feature) are completely **independent parameters** with no spatial relationship enforced.

```
What FC "sees":

  input[0]    →  ┐
  input[1]    →  │
  input[2]    →  │ → output[47]  (weighted mix of everything)
  ...         →  │
  input[25087]→  ┘

"I know index 0 exists. I know index 25087 exists.
 I have NO idea they were once neighbors in a 2D grid."
```

---

### Why This is Fatal for Segmentation

Segmentation needs to output **where** each class is — a spatial map. But after FC:

```
FC output: [0.9, 0.1, 0.05, 0.02, ...]
            ↑                        ↑
         "dog: 90%"           "cat: 2%"

This is ONE score for the WHOLE IMAGE.
There is no "dog is in the top-left, cat is in the bottom-right."
The location is gone forever.
```

To recover spatial information, you would need to:
1. Know WHICH input pixels were most responsible for each output
2. Map back from 25,088 flat indices to (i, j) positions in the 7×7 grid

But FC weights don't organize themselves spatially — the weight matrix $W$ treats all 25,088 inputs as a flat bag of numbers. **There's no way to reverse-engineer location from the output.**

---

### Contrast with Convolution — Spatial Info Preserved

A convolutional layer processes each position **independently and consistently**:

```
Conv 3×3 applied at position (2,3):
  → looks at a 3×3 neighborhood around (2,3)
  → output at (2,3) depends ONLY on inputs near (2,3)

Conv 3×3 applied at position (5,1):
  → looks at a 3×3 neighborhood around (5,1)
  → output at (5,1) depends ONLY on inputs near (5,1)
```

```
Feature map stays as 2D grid throughout:

  Input:  7×7×512
  Conv:   7×7×512  ← same spatial structure, location preserved
  Conv:   7×7×512
  Conv:   7×7×21   ← 21 classes, still 7×7 spatial grid
  
  Position (2,3) in output → corresponds to position (2,3) in input → top-center region
```

Location is never lost because the 2D structure is **never destroyed**.

---

### The Memory Analogy

Think of the feature map as a **city map** with labeled buildings:

```
Feature map (city map):
  ┌───────────────────────┐
  │ 🏥 hospital  │ 🏫 school│
  │              │         │
  │ 🏪 shop      │ 🏠 house │
  └───────────────────────┘
  
  "hospital is top-left, school is top-right"
  → spatial relationship preserved
```

Flatten = **write all buildings in a list, left-to-right, top-to-bottom:**

```
[hospital, school, shop, house]
```

Now ask: "where is the hospital?" — **impossible to answer from the list alone** unless you remember it was originally at index 0 = top-left. The list carries no geometry.

FC layer = reads this list and outputs "there is a hospital" — but **cannot tell you it's in the top-left**. All location knowledge is gone.

---

**Interview-ready one-liner:**
> "Flattening converts a 2D spatial grid $(H \times W \times C)$ into a 1D vector of $H \times W \times C$ values, destroying the (i,j) coordinate system — a feature that was at position (2,3) is now just 'index 147' with no spatial meaning. The subsequent FC layer is a matrix multiply over all values jointly, so every output neuron is a weighted sum of the entire flattened input regardless of original location. This makes it impossible to produce spatially-organized outputs like pixel-level label maps — which is exactly why FCN replaces FC layers with convolutions that maintain the 2D structure throughout."

---

## U-Net: Symmetric Encoder-Decoder — Why It Preserves Spatial Info Better Than FCN

---

### The Core Problem with FCN

FCN's architecture is fundamentally **asymmetric**:

```
FCN (asymmetric):

Encoder:   5 deep stages (VGG-16 backbone) — rich, heavy
           ↓  ↓  ↓  ↓  ↓
Decoder:   1–2 shallow upsampling stages — thin, coarse
           ↑  ↑

Skip connections: 2–3 additive connections, late in decoder only
```

By the time the decoder starts reconstructing, most fine spatial detail (edges, exact boundaries, textures) has been **discarded** by pooling. The decoder tries to recover it from almost nothing. That's why FCN boundaries are blurry.

---

### U-Net's Symmetric Design

U-Net (Ronneberger et al., 2015) was designed for biomedical image segmentation where **exact pixel boundaries matter** (e.g., cell walls). The insight: *make the decoder just as deep and powerful as the encoder, and connect them at every resolution level*.

```
U-Net architecture (the actual "U" shape):

Encoder (contracting)              Decoder (expanding)
─────────────────────              ───────────────────
[Input 572×572]
Conv×2, 64ch  → [568×568×64] ────────────────────→ [Conv×2, 64ch] → [Output]
    MaxPool↓                                              TranspConv↑
[284×284×64]                                        ↑ concat ↑
Conv×2, 128ch → [280×280×128] ──────────────────→ [Conv×2, 128ch]
    MaxPool↓                                              TranspConv↑
[140×140×128]                                       ↑ concat ↑
Conv×2, 256ch → [136×136×256] ──────────────────→ [Conv×2, 256ch]
    MaxPool↓                                              TranspConv↑
[68×68×256]                                         ↑ concat ↑
Conv×2, 512ch → [64×64×512]  ───────────────────→ [Conv×2, 512ch]
    MaxPool↓                                              TranspConv↑
[32×32×512]                                         ↑ concat ↑
         ↘                                         ↗
           Conv×2, 1024ch — [28×28×1024] (bottleneck)
```

4 downsampling stages → 4 matching upsampling stages. Every level mirrors the encoder. That's the "symmetric" part.

---

### Skip Connections: Concatenation vs Addition (Critical Difference)

**FCN skip connections** use element-wise **addition**:

```
decoder_tensor   [H × W × C]
encoder_tensor + [H × W × C]  ← same shape required
─────────────────────────────
result           [H × W × C]  ← values blended together
```

Problem: If encoder says "edge here" = +5 and decoder says "no edge" = −5, they **cancel to 0**. Information is destroyed by interference.

---

**U-Net skip connections** use channel-wise **concatenation**:

```
decoder_tensor   [H × W × C]
encoder_tensor   [H × W × C]
─────────────────────────────
concat along ch  [H × W × 2C]  ← both preserved, side by side
                     ↓
              Conv 3×3 (learnable)  ← network decides how to fuse
```

Why concatenation wins:

1. **No signal cancellation** — both encoder and decoder features survive intact
2. **Learnable fusion** — the conv after concat learns which features matter per location
3. **Independent gradients** — backprop flows through both paths separately, stable training
4. **Double the information** — decoder simultaneously sees "what is here semantically" AND "where is the boundary spatially"

---

### What Each Path Carries at Each Scale

```
Scale      Encoder output contains       Decoder needs this for
─────      ────────────────────────      ──────────────────────
Stage 1    Fine edges, corners, noise    Exact boundary pixels
Stage 2    Mid-level texture, shapes     Cell walls, thin structures
Stage 3    Part-level features           Organ outlines, blobs
Stage 4    High-level context            "This whole region is nucleus"
Bottleneck Full semantic understanding   "Where to put the label"
```

FCN only connects skip connections at stages 3–4 (coarse). U-Net connects all 4. This means fine spatial detail from stage 1 **flows directly into the decoder's final upsampling step** — the decoder doesn't need to reconstruct edges from scratch.

---

### Transposed Convolution: Learnable Upsampling

FCN often uses fixed bilinear upsampling (no parameters). U-Net uses **transposed convolutions** — the upsampling kernel is *learned*.

For a 2×2 transposed conv with stride 2:

```
Input pixel  →  2×2 block output  (each input "broadcasts" to a patch)

[a  b]        [a  a  b  b]
[c  d]   →    [a  a  b  b]
              [c  c  d  d]
              [c  c  d  d]
   (overlapping regions summed with learned weights W)
```

Formally, transposed conv is the **gradient operator of a regular conv** — it's the "backward pass" of a strided convolution used as a forward operation.

**Known issue — checkerboard artifacts**: When kernel size is not divisible by stride, some output pixels receive contributions from more input pixels than others → periodic unevenness → grid pattern in output.

Fix used in modern U-Net variants:

```
❌ Transposed conv (stride=2, kernel=3)  — uneven overlap
✅ Bilinear upsample 2× (fixed) + Conv 3×3 (learned)
   → separates "scale up" from "refine" — no overlap artifacts
```

---

### FCN vs U-Net: Side-by-Side

| Property | FCN | U-Net |
|---|---|---|
| Architecture shape | Asymmetric (heavy encoder, light decoder) | Symmetric (equal depth both sides) |
| Skip connections | Additive (2–3 connections) | Concatenation (every level) |
| Decoder depth | Shallow | Matches encoder exactly |
| Upsampling | Fixed bilinear | Learnable transposed conv |
| Spatial preservation | Partial — coarse recovery | Strong — fine detail injected at each level |
| Boundary sharpness | Blurry | Sharp |
| Best for | Natural image classification tasks | Biomedical / fine boundary segmentation |

---

### Why "Symmetric" is the Right Word

In FCN, encoder:decoder depth ratio ≈ **5:1**. The decoder is an afterthought — a few upsampling steps tacked on.

In U-Net, encoder:decoder ratio = **4:4**. The decoder is equally principled. Every spatial resolution (568, 284, 140, 68) exists in *both* paths, and they're bridged by skip connections. The network has a symmetric opportunity to reason about features at every scale in both the downsampling and upsampling direction.

---

**Interview one-liner:**
> U-Net's symmetric decoder mirrors the encoder stage-for-stage, and concatenation-based skip connections inject fine spatial details at every resolution level — so the decoder never has to re-learn what boundaries look like from scratch, it already has them.

---

## U-Net Versatility: Why It Extends So Naturally to 3D and Beyond

---

### Why U-Net's Design is Inherently Generalizable

U-Net has one core architectural principle: **symmetric encoder-decoder with skip connections at every resolution level**. That principle is completely independent of:

- Whether input is 2D, 3D, or even 4D
- Whether the task is semantic labeling or instance separation
- Whether the domain is natural images, medical scans, or satellite imagery

This is what makes it a *framework* rather than a one-trick model.

---

## Extension 1: 3D U-Net for Volumetric Data

### What is Volumetric Data?

Medical imaging doesn't produce a single 2D slice — it produces a **stack of slices** that together form a 3D volume:

```
CT/MRI scan structure:

Slice 1:  ████████████
Slice 2:  ████████████   ← each slice is a 2D grayscale image
Slice 3:  ████████████
   ...        ...
Slice N:  ████████████

Together: a 3D tensor [D × H × W] (Depth × Height × Width)
```

If you segment each slice independently with a 2D U-Net, you lose **inter-slice continuity** — a tumor that spans slices 5–12 might be labeled inconsistently across slices because the model can't "see" that slices 5 and 6 are connected.

---

### What Changes in 3D U-Net

Every 2D convolution becomes a **3D convolution**:

```
2D Conv:  kernel [K × K]       operates on [H × W]
3D Conv:  kernel [K × K × K]   operates on [H × W × D]
```

A 3×3×3 kernel looks at a **3D neighborhood** — it can detect that a structure continues across adjacent slices.

The full architecture mirrors 2D U-Net but adds a depth dimension:

```
2D U-Net:                    3D U-Net:
─────────                    ────────
Input [H × W × C]            Input [D × H × W × C]
Conv2D + MaxPool2D           Conv3D + MaxPool3D
   ↓                              ↓
[H/2 × W/2 × 2C]            [D/2 × H/2 × W/2 × 2C]
   ↓                              ↓
Bottleneck                   Bottleneck
   ↑                              ↑
TranspConv2D + Concat        TranspConv3D + Concat
   ↑                              ↑
Output [H × W × num_classes] Output [D × H × W × num_classes]
```

Every input voxel (3D pixel) gets a class label. The skip connections now carry spatial information across all three axes — height, width, **and** depth.

---

### Why This Works: The Architecture Doesn't Care About Dimensionality

The skip connection logic is identical:

```
At resolution [D/2 × H/2 × W/2]:
  encoder output   [D/2 × H/2 × W/2 × C]
  decoder output   [D/2 × H/2 × W/2 × C]
  concat →         [D/2 × H/2 × W/2 × 2C]
  Conv3D →         fused 3D features
```

The network learns "this 3D neighborhood looks like a boundary" just as the 2D version learns "this 2D neighborhood looks like a boundary."

**Memory trade-off**: 3D volumes are large. A 512×512×256 MRI scan at float32 = **256 MB** just for the input. Activations at intermediate layers multiply this further. Solutions:

1. **Patch-based training** — train on small 3D crops (e.g., 128×128×64), infer with sliding window
2. **Anisotropic kernels** — if slice spacing ≠ pixel spacing, use [1×3×3] or [3×3×1] kernels
3. **2.5D approach** — take 3 adjacent slices as 3-channel 2D input (cheaper, but less accurate)

---

## Extension 2: U-Net for Instance Segmentation

### The Gap: Semantic vs Instance

Standard U-Net outputs a class label per pixel:

```
Semantic output: "every pixel labeled — but all cells labeled 'cell'"

Input image:        Semantic output:
  ○  ○  ○            1  1  1
  ○  ○  ○    →       1  1  1     (all cells = class 1, background = 0)
  ○  ○  ○            1  1  1

Problem: can't tell where one cell ends and another begins
```

Instance segmentation needs: "cell 1 here, cell 2 there, cell 3 over there."

---

### Strategy 1: Distance Transform Head (used in Cell U-Net)

Add a second output head that predicts, for each pixel, **how far it is from the nearest boundary**:

```
U-Net encoder/decoder backbone
          ↓
    shared features
    ↙            ↘
Head 1:          Head 2:
Semantic mask    Distance transform
(foreground/bg)  (center=high, edge=low)

Combined:
  - Find local maxima in distance map → instance centers
  - Watershed from those centers → instance boundaries
```

The distance transform creates a "hill" for each object — centers are high, boundaries are zero. Watershed flooding from the peaks naturally separates touching instances.

```
Distance transform example (top view):

   0 0 0 0 0 0 0 0
   0 1 1 1 0 1 1 0
   0 1 2 1 0 1 2 0   ← peak=2 means "center of instance"
   0 1 1 1 0 1 1 0
   0 0 0 0 0 0 0 0

Two peaks → two instances, watershed separates them at the valley (0s)
```

---

### Strategy 2: Contour/Border Prediction (U-Net++)

Add a third head that predicts **touching borders** explicitly:

```
Head 1: Foreground mask
Head 2: Instance border mask  ← explicitly marks where instances touch
Head 3: (optional) distance

Post-process:
  Foreground ∩ NOT border → individual instance regions
  Connected components → unique instance IDs
```

---

### Strategy 3: Embedding U-Net (for arbitrary instance counts)

Each pixel predicts an **embedding vector** such that pixels from the same instance cluster together and pixels from different instances are far apart:

```
U-Net output: [H × W × D]  where D = embedding dimension (e.g., 32)

Pixel at (i,j) outputs vector v_ij ∈ ℝ^32

Training loss (discriminative loss):
  - Pull: pixels of same instance → their embeddings should be close
  - Push: pixels of different instances → embeddings should be far apart

Inference: cluster embeddings → each cluster = one instance
```

This handles **arbitrary numbers of instances** without any anchor or proposal mechanism.

---

## Other Extensions

| Extension | What changes | Use case |
|---|---|---|
| **3D U-Net** | All convs/pools become 3D | CT, MRI, electron microscopy |
| **Attention U-Net** | Add attention gates on skip connections — suppress irrelevant regions | Abdominal organ seg (suppress background) |
| **U-Net++** | Nested dense skip connections — intermediate nodes between encoder and decoder | Better feature reuse, easier pruning |
| **ResU-Net** | Replace conv blocks with residual blocks | Deeper networks without vanishing gradients |
| **TransUNet** | Hybrid: CNN encoder + Transformer bottleneck + CNN decoder | Long-range dependencies + spatial precision |
| **nnU-Net** | Auto-configures all U-Net hyperparameters based on dataset stats | Any medical segmentation task, state-of-the-art out of box |

---

### Why U-Net is the Right Base

The reason U-Net extends so naturally is that its two core mechanisms are **domain-agnostic**:

1. **Hierarchical feature encoding** — works on any structured spatial data (2D, 3D, n-D)
2. **Skip connections restoring spatial detail** — the problem "coarse features lose fine detail" exists in every spatial prediction task, not just 2D images

The result: U-Net is arguably the most widely adopted architecture in medical image analysis, and its design patterns appear in nearly every modern segmentation model.

---

**Interview one-liner:**
> U-Net's symmetric encoder-decoder with skip connections is dimension-agnostic — replacing 2D ops with 3D ops directly produces a volumetric segmentation network, and adding auxiliary output heads (distance transform, border prediction, pixel embeddings) extends it to instance segmentation, making it a framework rather than a fixed model.

---

## Anisotropic Kernels: When Slice Spacing ≠ Pixel Spacing

---

### The Problem: Medical Volumes Are Not Isotropic

In natural images, every pixel represents the same physical size. A pixel is a pixel.

In medical imaging, this is **often not true**. The spatial resolution differs between the in-plane directions (H, W) and the through-plane direction (D/depth/slice axis):

```
CT scan example:
  In-plane pixel spacing:   0.5 mm × 0.5 mm   (fine — detector resolution)
  Slice thickness:          5.0 mm             (coarse — patient movement, dose limits)

So one "voxel" physically represents:
  0.5 mm × 0.5 mm × 5.0 mm   ← a tall, thin slab, not a cube
```

This means adjacent slices (depth neighbors) are **10× further apart** in physical space than adjacent pixels within a slice.

---

### Why This Breaks a Cubic [3×3×3] Kernel

A standard 3×3×3 kernel assumes all 3 axes are equally spaced — it treats depth neighbors and in-plane neighbors as equally "close":

```
[3×3×3] kernel neighborhood:

      Slice z-1       Slice z        Slice z+1
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │  x x x   │  │  x x x   │  │  x x x   │
   │  x x x   │  │  x C x   │  │  x x x   │  C = center voxel
   │  x x x   │  │  x x x   │  │  x x x   │
   └───────────┘  └───────────┘  └───────────┘

This kernel says: "the 8 neighbors in slice z±1 are just as
relevant as the 8 neighbors within slice z"
```

But physically, the neighbors in slice z±1 are 5 mm away, while in-plane neighbors are only 0.5 mm away. The kernel is **mixing features from very different physical scales** — it's like treating a neighbor 1 meter away the same as a neighbor 10 meters away.

Consequences:
1. **Blurring across slices** — fine structures that fit within a single slice get smeared across adjacent slices
2. **Wasted parameters** — the depth-direction kernels are learning relationships between voxels that aren't spatially coherent
3. **Boundary artifacts** — object boundaries that are sharp within a slice appear diffuse in the depth direction

---

### The Fix: Anisotropic Kernels Match the Physical Anisotropy

**Option A: [1×3×3] kernel** — operates only within a single slice

```
[1×3×3] kernel — no depth dimension at all:

      Slice z-1       Slice z        Slice z+1
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │           │  │  x x x   │  │           │
   │  (empty)  │  │  x C x   │  │  (empty)  │  ← only current slice
   │           │  │  x x x   │  │           │
   └───────────┘  └───────────┘  └───────────┘

Depth neighbors: ignored entirely
In-plane neighbors: full 3×3 neighborhood
```

Use this when slices are very thick and essentially independent. Each slice is processed as if it's 2D. You get the full spatial reasoning within each slice without any cross-slice contamination.

---

**Option B: Pseudo-3D — [1×3×3] + [3×1×1] in sequence**

```
Instead of one [3×3×3] kernel, use two kernels in sequence:

Step 1: [1×3×3]  — learn in-plane features (fine resolution)
Step 2: [3×1×1]  — learn cross-slice features (coarse resolution)

This is called "pseudo-3D" or "anisotropic factorization"
```

---

### Numerical Intuition

Imagine a lung nodule in a CT scan:

```
Physical extent of a 5mm nodule:
  In-plane:  5mm / 0.5mm = 10 pixels diameter  (well-resolved)
  Depth:     5mm / 5.0mm = 1 slice thickness    (barely 1 voxel)
```

With a cubic [3×3×3] kernel:
- The depth receptive field covers 3 × 5mm = **15mm** of real tissue
- The in-plane receptive field covers 3 × 0.5mm = **1.5mm** of real tissue
- The kernel is looking at a 15mm depth range to inform a 1.5mm spatial prediction — wildly mismatched

With [1×3×3]:
- Depth receptive field: **0mm** (stays within one slice)
- In-plane receptive field: **1.5mm** — appropriate for fine in-plane detail

---

### Pooling Is Affected Too

The same logic applies to max pooling and transposed convolution:

```
Isotropic (voxel spacing ≈ equal):
  MaxPool3D(2, 2, 2) — halve all three axes equally

Anisotropic (slices much thicker):
  MaxPool3D(1, 2, 2) — only halve H and W, leave D alone

  Why? If you halve D when slices are already thick, you lose
  the small amount of depth information you have entirely.
  Better to preserve depth resolution and only compress
  the fine in-plane axes.
```

The encoder then gradually starts pooling the depth axis only once the in-plane resolution has been reduced enough that the relative anisotropy decreases:

```
Stage 1 pool: (1, 2, 2)  → [D × H/2 × W/2]
Stage 2 pool: (1, 2, 2)  → [D × H/4 × W/4]
Stage 3 pool: (2, 2, 2)  → [D/2 × H/8 × W/8]   ← depth pooling starts here
```

---

### How nnU-Net Handles This Automatically

nnU-Net computes the voxel spacing from the dataset metadata and automatically selects kernel shapes:

```python
# Pseudocode of nnU-Net's kernel selection logic:

if max_spacing / min_spacing > 2:
    # Anisotropic dataset
    kernel = [1, 3, 3]
    pool_kernel = [1, 2, 2]   # don't pool depth axis yet
else:
    # Isotropic dataset
    kernel = [3, 3, 3]
    pool_kernel = [2, 2, 2]   # pool all axes equally
```

This is why nnU-Net achieves state-of-the-art results out-of-the-box — it respects the physical geometry of the data.

---

### Summary Table

| Scenario | Physical spacing | Kernel choice | Reason |
|---|---|---|---|
| Isotropic MRI (1×1×1 mm) | Equal all axes | [3×3×3] | All neighbors equally relevant |
| CT with thick slices (0.5×0.5×5 mm) | Depth 10× coarser | [1×3×3] or [1×3×3]+[3×1×1] | Depth neighbors are far away |
| Ultrasound (1×1×2 mm) | Slightly anisotropic | [3×3×3] or mild anisotropic | Small difference, cubic usually fine |
| Histology stack (0.25×0.25×10 mm) | Depth 40× coarser | [1×3×3] strictly | Slices essentially independent |

---

**Interview one-liner:**
> Medical volumes are physically anisotropic — in-plane pixels may be 0.5 mm apart while slices are 5 mm apart, so a cubic [3×3×3] kernel treats spatially distant depth-neighbors as equally relevant as close in-plane neighbors. Anisotropic kernels like [1×3×3] restrict convolution to axes where voxels are actually spatially coherent, preventing cross-scale feature blending and matching the kernel's receptive field to the real-world geometry of the data.

---

## U-Net for Satellite Image Analysis

---

### Why Satellite Imagery is a Natural Fit for U-Net

Satellite images share the core challenge that U-Net was designed to solve: **dense, pixel-level prediction over large spatial areas where both fine detail and broad context matter simultaneously**.

```
Satellite image characteristics:
  - Very high resolution: 0.3m–30m per pixel (WorldView-3 to Landsat)
  - Multi-spectral: RGB + NIR + SWIR + thermal bands (4–13 channels)
  - Large image size: 10,000×10,000+ pixels per tile
  - Objects of interest: roads (1–5px wide), buildings (10–100px), forests (millions of px)
  - All require: "what is each pixel?" — exactly what U-Net does
```

---

### Core Satellite Segmentation Tasks

#### Task 1: Building Footprint Detection

```
Input: RGB aerial/satellite image

Target output (per pixel):
  0 = background / road / vegetation
  1 = building footprint

Challenge: buildings touch each other, share walls, vary in size
from 3×3 px (shed) to 500×500 px (warehouse)
```

U-Net's multi-scale skip connections handle this naturally:
- **Coarse features** (bottleneck): "this is a dense urban area"
- **Fine features** (early encoder): "this specific edge at pixel (i,j) is a wall"

A single-scale model misses either the urban context or the fine boundary.

---

#### Task 2: Road Network Extraction

Roads are among the hardest satellite targets:

```
Challenge: roads are thin (1–5px wide in high-res imagery)
but can span thousands of pixels in length

Standard convnet:
  After 4× MaxPool → feature map is 1/16 size
  A 3px road → 0.19px in feature space → VANISHES

U-Net fix:
  Skip connection from stage 1 (before pooling) carries
  the full-resolution road pixels directly to the decoder
  → thin structures survive the encoder's spatial compression
```

Additional technique: **binary cross-entropy + distance transform weighting** — pixels near road centerlines get higher loss weight so the network learns precise centerlines rather than blurry blobs.

---

#### Task 3: Land Cover / Land Use Classification

```
Classes: water, forest, cropland, urban, bare soil, snow/ice

Multi-class per-pixel prediction:
  Input:  [H × W × 13]  (Sentinel-2 has 13 spectral bands)
  Output: [H × W × 6]   (6 land cover classes, softmax)

U-Net advantage: captures both
  - Local texture (forest texture vs cropland texture) — fine features
  - Regional context (river valley vs plateau) — coarse features
```

---

#### Task 4: Change Detection

Two images of the same location at different times → predict which pixels changed:

```
Input:  Image_t1 [H × W × C]  +  Image_t2 [H × W × C]
Output: Change mask [H × W × 1]  (0=no change, 1=change)

Siamese U-Net architecture:
  Encoder_t1  (shared weights)  →  features_t1
  Encoder_t2  (shared weights)  →  features_t2
                                        ↓
               Difference or concat at each skip level
                                        ↓
                          Shared decoder → change map
```

Sharing encoder weights forces the model to extract comparable features from both time points. Skip connections carry fine temporal differences (a new building's exact edge) alongside broad contextual change signals.

---

### Satellite-Specific Challenges and U-Net Adaptations

#### Challenge 1: Images Are Too Large to Process Whole

A 10,000×10,000 satellite image at float32 with 13 bands = **5.2 GB** just for the input. U-Net cannot process this in one pass.

**Solution: Sliding window with overlap-tile strategy** (from the original U-Net paper):

```
Tile the image into overlapping patches:

┌──────┬──────┬──────┐
│  P1  │  P2  │  P3  │
├──────┼──────┼──────┤   Each patch: 512×512 px
│  P4  │  P5  │  P6  │   Overlap: 64px on each side
├──────┼──────┼──────┤
│  P7  │  P8  │  P9  │
└──────┴──────┴──────┘

Process each patch → predictions
Stitch predictions → discard border overlap (border predictions
are less reliable due to missing context at patch edges)
```

The overlap ensures that even pixels at patch boundaries have full local context from both neighboring patches.

---

#### Challenge 2: Multi-Spectral Input (More Than 3 Channels)

Standard U-Net uses RGB (3 channels). Satellite sensors provide many more:

```
Sentinel-2 bands:
  Band 2 (Blue):    490nm
  Band 3 (Green):   560nm
  Band 4 (Red):     665nm
  Band 8 (NIR):     842nm   ← vegetation health
  Band 11 (SWIR):   1610nm  ← moisture, soil
  Band 12 (SWIR2):  2190nm  ← mineral discrimination
  ... (13 bands total)

Derived indices often added as extra channels:
  NDVI = (NIR - Red) / (NIR + Red)   ← vegetation index
  NDWI = (Green - NIR) / (Green + NIR)  ← water index
```

U-Net handles this trivially — the first conv layer just changes from 3 input channels to N:

```python
# Standard U-Net first conv:
Conv2d(in_channels=3,  out_channels=64, kernel_size=3)

# Satellite U-Net first conv:
Conv2d(in_channels=13, out_channels=64, kernel_size=3)  # same architecture
```

The network learns which spectral bands are informative for each class entirely from data.

---

#### Challenge 3: Class Imbalance is Severe

```
Typical land cover distribution in a satellite scene:
  Forest:      45%
  Cropland:    30%
  Water:       12%
  Urban:        8%
  Buildings:    3%
  Roads:        2%
```

Roads and buildings (most commercially important) are rare. Solutions:

1. **Weighted cross-entropy**: rare classes get higher loss multiplier
$$\mathcal{L} = -\sum_c w_c \cdot y_c \log(\hat{y}_c)$$
where $w_c \propto 1/\text{frequency}_c$

2. **Dice loss**: directly optimizes overlap, inherently balances classes
$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum y \cdot \hat{y}}{\sum y + \sum \hat{y}}$$

3. **Focal loss** (widely adopted for satellite):
$$\mathcal{L}_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)$$
Down-weights easy background pixels, focuses training on hard road/building pixels

---

#### Challenge 4: Varying Spatial Resolution Across Sensors

```
Sensor          GSD        Typical use
──────          ───        ───────────
WorldView-3     0.3m       Building footprints, fine detail
Pleiades        0.5m       Urban mapping
Sentinel-2      10m        Land cover, agriculture
Landsat-8       30m        Regional vegetation, water bodies
MODIS           250m–1km   Global monitoring
```

The same model architecture works across sensors — **transfer learning + fine-tuning** is standard: pre-train on high-data sensor, fine-tune on target sensor.

---

#### Challenge 5: Temporal and Seasonal Variation

A forest in summer vs winter looks completely different spectrally. Solutions:
1. **Spectral jitter augmentation** — randomly perturb band values during training
2. **Multi-temporal U-Net** — stack images from multiple dates as extra channels
3. **ConvLSTM decoder** — process a time series, decoder uses recurrent connections

---

### Benchmark Datasets for Satellite U-Net

| Dataset | Task | GSD |
|---|---|---|
| SpaceNet (1–8) | Building footprints, roads | 0.3m–0.5m |
| ISPRS Vaihingen/Potsdam | Urban land cover | 0.05m–0.09m |
| DeepGlobe | Road, building, land cover | 0.5m |
| iSAID | Instance segmentation | Various |
| BigEarthNet | Multi-label land cover (Sentinel-2) | 10m |

U-Net and its variants (ResU-Net, Attention U-Net, TransUNet) consistently top leaderboards on all of these.

---

### Why U-Net Specifically (vs Generic Segmentation Models)

| Property | Why it matters for satellite |
|---|---|
| No FC layers | Works on arbitrary image sizes — critical for large tiles |
| Skip connections at every level | Thin roads (fine) + land cover regions (coarse) both need attention |
| Lightweight — trains on limited labels | Satellite labels are expensive |
| Easily extended to N input channels | Multi-spectral bands drop in without architecture changes |
| Tiling-friendly | Overlap-tile strategy works out-of-box |

---

**Interview one-liner:**
> Satellite imagery demands exactly what U-Net delivers — per-pixel labels over large scenes where thin roads (1–3px) and vast forests coexist. U-Net's multi-scale skip connections preserve fine linear structures while coarse encoder features capture regional land cover context, and the fully convolutional design handles arbitrary-size tiles from any multi-spectral sensor without architecture changes.

---

## Mask R-CNN: Two-Stage Instance Segmentation

---

### Starting Point: What Faster R-CNN Does

To understand Mask R-CNN, you must first understand what it extends. Faster R-CNN produces **bounding boxes + class labels** for each detected object in two stages:

```
Stage 1 — Region Proposal Network (RPN):
  Input image → CNN backbone → feature map
  RPN slides over feature map → proposes ~2000 candidate boxes
  ("something is here" — class-agnostic)

Stage 2 — Detection Head:
  For each proposal → crop the feature map → classify + refine box
  Output: [class_label, x1, y1, x2, y2] per object
```

Mask R-CNN asks: **what if we added a third output — a pixel mask — alongside the box and class label?**

---

### The Mask R-CNN Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Backbone + FPN                │
                    │  (ResNet-50/101 + Feature Pyramid Net)  │
                    └──────────────┬──────────────────────────┘
                                   │ feature maps (multi-scale)
                    ┌──────────────▼──────────────┐
                    │    Region Proposal Network   │
                    │  → ~2000 region proposals    │
                    └──────────────┬──────────────┘
                                   │ top-K proposals (e.g., 300)
                    ┌──────────────▼──────────────┐
                    │         RoIAlign             │  ← KEY upgrade from Faster R-CNN
                    │  crops + aligns feature map  │
                    │  to fixed 7×7 or 14×14 grid  │
                    └──────┬───────────┬──────────┘
                           │           │
              ┌────────────▼──┐   ┌────▼────────────┐
              │  Box Head      │   │   Mask Head      │
              │  FC layers     │   │   Conv layers    │
              │  → class score │   │  → binary mask   │
              │  → box offset  │   │  [28×28 per obj] │
              └────────────────┘   └─────────────────┘
```

Three parallel output heads, all running on the same RoIAlign feature crop:
1. **Classification**: what class is this object?
2. **Box regression**: refine the bounding box coordinates
3. **Mask prediction**: predict a pixel-level binary mask for this object

---

### Stage 1: Backbone + FPN

Mask R-CNN uses **Feature Pyramid Network (FPN)** on top of ResNet:

```
ResNet stages:         Feature maps:       FPN output:
                                           (multi-scale)
Input image
    ↓
  C1 [H/2]
  C2 [H/4]   ─────────────────────────→  P2 [H/4,  256ch]  ← fine detail
  C3 [H/8]   ─────────────────────────→  P3 [H/8,  256ch]
  C4 [H/16]  ─────────────────────────→  P4 [H/16, 256ch]
  C5 [H/32]  ─────────────────────────→  P5 [H/32, 256ch]  ← coarse semantics
                          ↑
               top-down pathway: upsample
               and add lateral connections
```

Each proposal is assigned to the FPN level based on its size:

$$k = k_0 + \lfloor \log_2(\sqrt{wh} / 224) \rfloor$$

where $k_0 = 4$, $w$ and $h$ are proposal width and height.

---

### Stage 2: RoIAlign — The Critical Upgrade

#### The RoIPool Quantization Problem

Faster R-CNN uses **RoIPool** which rounds coordinates to integer grid positions:

```
Proposal box in image space: [x1=12.3, y1=5.7, x2=47.8, y2=38.2]

Step 1 — map to feature space (stride=16):
  x1 = floor(12.3/16) = floor(0.77) = 0   ← QUANTIZATION ERROR
  y1 = floor(5.7/16)  = floor(0.36) = 0
  x2 = floor(47.8/16) = floor(2.99) = 2
  y2 = floor(38.2/16) = floor(2.39) = 2

Problem: the floor() rounding misaligns the feature crop from the
actual proposal location.
```

For **bounding box detection**, small misalignment doesn't matter much. For **pixel-level masks**, it's catastrophic — a 1-pixel misalignment at stride 16 = a 16-pixel error in image space.

---

#### RoIAlign: Bilinear Interpolation, No Quantization

RoIAlign **never quantizes**. It keeps all coordinates as floating-point and uses bilinear interpolation to sample feature values:

```
Within each bin, sample at 4 fixed points (2×2 sampling grid):
  Sample point at (1.1, 0.7) → bilinear interpolation from
  the 4 surrounding feature map pixels

  value = w_tl * f[0,0] + w_tr * f[0,1]
        + w_bl * f[1,0] + w_br * f[1,1]

  where weights w = (1-dx)(1-dy), (1-dx)dy, dx(1-dy), dx·dy
  and dx, dy are fractional distances to integer grid points
```

```
Feature map (integer grid):

  ·    ·    ·    ·

  ·    ·    ·    ·
       ×              ← sample point at (1.3, 1.7) — NOT on integer grid
  ·    ·    ·    ·    ← bilinear interpolation from surrounding 4 pixels

  ·    ·    ·    ·
```

Result: **exact, sub-pixel accurate feature extraction** — the cropped feature map is perfectly aligned with the proposal location.

---

### Stage 2: The Mask Head

After RoIAlign produces a 14×14 feature crop per proposal:

```
RoIAlign output: [14×14×256]  (per proposal)
     ↓
Conv 3×3, 256ch  → ReLU
Conv 3×3, 256ch  → ReLU
Conv 3×3, 256ch  → ReLU
Conv 3×3, 256ch  → ReLU   (4 conv layers)
     ↓
TranspConv 2×2, stride=2  → upsample to [28×28×256]
     ↓
Conv 1×1, num_classes     → [28×28×K]  (K binary masks, one per class)
     ↓
Sigmoid activation        → [28×28×K] ∈ [0,1]
```

**Key design decision: K independent binary masks, not a single K-class softmax.**

Why? Decouples mask prediction from classification. The mask head just asks "is this pixel part of the object?" for each class independently. The class label comes from the box head. At inference: use the predicted class to select which of the K masks to output.

---

### Training: Multi-Task Loss

$$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{mask}$$

- $\mathcal{L}_{cls}$ — cross-entropy over class predictions
- $\mathcal{L}_{box}$ — smooth L1 loss on box coordinate regression
- $\mathcal{L}_{mask}$ — average binary cross-entropy over the 28×28 mask, computed **only for the ground-truth class**:

$$\mathcal{L}_{mask} = -\frac{1}{28^2} \sum_{i,j} \left[ y_{ij} \log \hat{m}_{ij}^k + (1-y_{ij}) \log(1 - \hat{m}_{ij}^k) \right]$$

where $k$ = ground-truth class, $y_{ij} \in \{0,1\}$ = ground-truth mask pixel.

Only **foreground proposals** (IoU > 0.5 with a GT box) contribute to mask loss.

---

### Why Two Stages? What Does Each Stage Buy?

```
Stage 1 (RPN):
  - Finds WHERE objects are (rough localization)
  - Class-agnostic — just "something is here"
  - Filters out 99% of image as background
  - Gives Stage 2 a manageable set of good candidates

Stage 2 (Detection + Mask heads):
  - Classifies WHAT each candidate is
  - Precisely localizes the box
  - Predicts fine-grained mask within the box

Why not one stage?
  One-stage models (YOLO, SSD) struggle with accurate masks
  because they never have a refined, class-specific crop to
  predict the mask from. The two-stage refinement gives the
  mask head a well-localized, properly-sized feature region.
```

---

### High Accuracy vs Slower Inference

| Metric | Mask R-CNN | One-stage (e.g., YOLACT) |
|---|---|---|
| COCO mask AP | ~37–39% | ~29–31% |
| Inference speed | ~5 FPS (ResNet-101) | ~30+ FPS |
| Small object accuracy | High (FPN helps) | Lower |
| Mask quality | Sharp, well-aligned | Coarser |
| Why slower | Two forward passes + RoIAlign per proposal | Single pass, no per-proposal ops |

**Optimizations used in practice:** reduce proposals to 300 at test time, FP16 inference, TensorRT, smaller backbone (ResNet-50).

---

### Mask R-CNN Variants and Extensions

| Variant | What it adds |
|---|---|
| **Mask R-CNN + keypoints** | Third head predicts 17 human keypoints — same RoIAlign crop |
| **PANet** | Bottom-up pathway added to FPN, improves small object masks |
| **Cascade Mask R-CNN** | Three sequential refinement stages with increasing IoU thresholds |
| **HTC** (Hybrid Task Cascade) | Interleaves detection and segmentation heads, feeds mask features back |
| **PointRend** | Iterative point-based mask rendering — much sharper boundaries |

---

**Interview one-liner:**
> Mask R-CNN extends Faster R-CNN by adding a parallel fully-convolutional mask head that runs on RoIAlign-extracted crops — RoIAlign's bilinear interpolation eliminates the quantization misalignment of RoIPool, giving pixel-accurate mask predictions, while K independent binary masks per class decouple segmentation from classification, letting each head specialize. The two-stage design (RPN → refine) gives the mask head a well-localized crop to work from, which is why it achieves higher mask accuracy than one-stage approaches at the cost of per-proposal computation.

---

## YOLACT: One-Stage Instance Segmentation via Linear Mask Assembly

---

### The Core Problem YOLACT Solves

Mask R-CNN is accurate but slow because it runs a **separate mask head for every proposed object**. For 300 proposals, you run 300 forward passes through the mask head.

YOLACT's key insight: **don't predict masks directly per object — factor mask prediction into two globally cheap operations that compose together**.

The idea is borrowed from signal processing: any complex signal can be approximated as a **linear combination of basis functions**. YOLACT applies this to masks:

```
Instance mask = c₁ · P₁ + c₂ · P₂ + c₃ · P₃ + ... + cₖ · Pₖ

where:
  Pᵢ = prototype masks (basis functions, shared across all instances)
  cᵢ = mask coefficients (per-object weights, predicted per detection)
```

Generate the prototypes **once** for the whole image. Predict coefficients **per object** (cheap — just K numbers per box). Assemble masks via matrix multiply. Done.

---

### Step 1: Prototype Masks — The Shared Basis

A **prototype mask** is a full-resolution soft mask that encodes some spatial pattern. They are not tied to any specific object or class — they are reusable spatial "templates".

```
Prototype mask examples (conceptual):

P₁: high values on left half of image
  ████████░░░░░░░░
  ████████░░░░░░░░
  ████████░░░░░░░░

P₂: high values in center
  ░░░░░░░░░░░░░░░░
  ░░░░████████░░░░
  ░░░░░░░░░░░░░░░░

P₃: high values on right, bottom
  ░░░░░░░░░░░░░░░░
  ░░░░░░░░████████
  ░░░░░░░░████████

P₄: diagonal gradient
  ████░░░░░░░░░░░░
  ░░████░░░░░░░░░░
  ░░░░████░░░░░░░░
```

These are **learned** — not hand-designed. The network discovers what spatial patterns are most useful as bases for reconstructing instance masks on the training set.

---

### How Prototypes Are Generated: The Protonet

A small FCN (called **Protonet**) runs once on the full image and outputs K prototype masks:

```
Input image [H × W × 3]
      ↓
  FPN backbone (shared with detection head)
      ↓
  Protonet (FCN, runs on P3 — finest FPN level):
    Conv 3×3, 256ch → ReLU
    Conv 3×3, 256ch → ReLU
    Conv 3×3, 256ch → ReLU
    TranspConv 2× upsample
    Conv 1×1, K channels  (K = 32 in YOLACT)
      ↓
  Output: [H/4 × W/4 × K]   (prototype maps at 1/4 image resolution)
```

K = 32 prototypes is enough. **Crucially: this runs ONCE per image, not once per object.** Cost is O(1) regardless of how many objects are in the scene.

---

### Step 2: Mask Coefficients — Per-Object Weights

YOLACT builds on a one-stage detector (specifically **RetinaNet** with FPN). The detection head predicts, for each anchor:

```
Standard detection outputs (per anchor):
  - Class score (C values)
  - Box offset (4 values: Δx, Δy, Δw, Δh)

YOLACT adds:
  - Mask coefficients (K values: c₁, c₂, ..., cₖ)  ← new head
```

The K coefficients are passed through **tanh** (not sigmoid or softmax):

```python
coefficients = tanh(raw_coeff_output)   # ∈ [-1, +1] per coefficient
```

Why tanh? Masks need **signed** combinations — some prototypes might need to be *subtracted* to carve out precise object boundaries. A sigmoid would restrict to positive combinations only, losing expressiveness.

---

### Step 3: Mask Assembly — Linear Combination

After NMS selects the final detections, mask assembly is a single matrix multiply:

```
Let:
  P = prototype matrix   [H/4 × W/4 × K]   (one set for the whole image)
  C = coefficient matrix [N × K]            (N = number of detected objects)

Assembly:
  M = sigmoid(P · Cᵀ)   [H/4 × W/4 × N]

where:
  P · Cᵀ  =  matrix multiply  →  [H/4 × W/4 × N]
  sigmoid →  soft mask per instance ∈ [0,1]
  threshold at 0.5 → binary mask

Final step: crop M to each object's predicted bounding box
  (removes false positives outside the box region)
```

Visualized:

```
Prototype P₁ [H×W]:   ████░░░░   (left-biased)
Prototype P₂ [H×W]:   ░░░░████   (right-biased)
Prototype P₃ [H×W]:   ░░████░░   (center)

Object A coefficients: [+0.8, -0.2, +0.5]

Mask_A = sigmoid(0.8·P₁ - 0.2·P₂ + 0.5·P₃)
       = sigmoid(high on left + slight suppression right + center boost)
       = a mask that covers the left-center region → fits object A
```

---

### Full YOLACT Architecture

```
Input image [550×550]
      ↓
ResNet-101 + FPN → {P3, P4, P5, P6, P7}   (shared backbone)
      ↓                        ↓
  Protonet                 Detection Head
  (runs on P3)             (runs on all FPN levels)
      ↓                        ↓
  [138×138×32]             Per anchor:
  prototype maps           - class scores (80 classes)
                           - box offsets (4)
                           - mask coeffs (32)  ← tanh activated
      ↓                        ↓
                        NMS → top N detections
                               ↓
                    ┌─────────────────────┐
                    │   Mask Assembly     │
                    │ M = sigmoid(P · Cᵀ) │
                    │  [138×138×N]        │
                    └─────────────────────┘
                               ↓
                    Crop to bounding box + resize
                               ↓
                    Final instance masks [H×W×N]
```

---

### Why YOLACT Is Fast: Cost Analysis

```
Operation            Mask R-CNN              YOLACT
─────────────────    ──────────────────      ───────────────────
Backbone + FPN       once                    once
Region proposals     RPN (separate pass)     none — anchor-based
Feature extraction   RoIAlign × 300 crops    none
Mask prediction      Mask head × 300 times   Protonet × 1 time
Mask assembly        none (direct output)    matrix multiply × 1

Total mask cost:     O(N) per-proposal ops   O(1) protonet + O(N·K) matmul
```

**Real-world speed:**
```
Model                   FPS (Titan Xp)    COCO mask AP
──────────────────────  ──────────────    ────────────
Mask R-CNN (ResNet-101) ~5 FPS            35.7%
YOLACT (ResNet-101)     ~30 FPS           31.2%
YOLACT-550 (ResNet-50)  ~42 FPS           28.2%
YOLACT++ (ResNet-50)    ~33 FPS           34.1%
```

~6× faster than Mask R-CNN with only ~4% AP drop.

---

### Why Accuracy Is Slightly Lower: The Limitations of Linear Combinations

**Problem 1: Prototype entanglement for nearby objects**

```
Two people standing side by side:

Person A:   ████░░░░
Person B:   ░░░░████

A single prototype might activate for both:
P₁:  ████████   (broad activation across both)

Coefficients:
  Person A: c₁ = +1.0
  Person B: c₁ = +1.0

Linear combination can't separate them without additional
prototypes that specifically distinguish left vs right
```

Mask R-CNN's per-object crop sidesteps this by predicting the mask in a **local coordinate frame** centered on each object.

**Problem 2: Mask leakage across instances**

The cropping-to-bounding-box step prevents mask leakage, but if bounding boxes overlap significantly, the crop doesn't fully isolate the instance.

**Problem 3: Low prototype resolution**

At 1/4 input resolution (138×138 for 550×550 input), fine details smaller than ~4 pixels are lost.

---

### YOLACT++ Improvements

1. **Mask re-scoring**: predict mask quality (IoU between predicted mask and GT), multiply final confidence — better ranking
2. **Deformable convolutions** in backbone: better spatial alignment for irregular shapes
3. **Optimized NMS**: faster post-processing

Result: YOLACT++ at 33 FPS achieves 34.1% mAP — competitive with Mask R-CNN's 35.7% at 5 FPS.

---

### YOLACT vs Mask R-CNN: Design Philosophy

| Dimension | Mask R-CNN | YOLACT |
|---|---|---|
| Mask prediction | Per-object local crop | Global prototypes + per-object coefficients |
| Speed | Slow (O(N) mask heads) | Fast (O(1) protonet + cheap assembly) |
| Accuracy | Higher (local coordinate frame) | Slightly lower (global basis) |
| Nearby instance separation | Strong | Weaker |
| Architecture style | Two-stage | One-stage |
| Real-time capable | No | Yes |
| Key innovation | RoIAlign + per-class binary masks | Prototype factorization |

---

**Interview one-liner:**
> YOLACT factors instance mask prediction into two cheap global operations: a Protonet generates K=32 shared spatial prototype masks once per image, and the detection head predicts K coefficients per object. Masks are assembled via a single matrix multiply — `sigmoid(P · Cᵀ)` — making mask generation O(1) regardless of object count. Accuracy is slightly lower than Mask R-CNN because the linear combination basis struggles to separate nearby same-class instances without a per-object local coordinate frame.

---

*End of notes — continued in next session.*
