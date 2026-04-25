# Computer Vision — Deeper Conceptual Notes

> Companion file to `foundationcv.md`. Contains in-depth Q&A with full equations and conceptual breakdowns.

---

## Q: What is the output size of a convolution?

### Formula

$$
\text{Output Size} = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1
$$

| Symbol | Meaning |
|--------|---------|
| $I$ | Input size (height or width) |
| $K$ | Kernel (filter) size |
| $P$ | Padding added on each side |
| $S$ | Stride |

> The formula applies independently to height and width. For non-square inputs, compute each separately.

---

### What Each Parameter Controls

**Padding ($P$)**
- Adds rows/columns of zeros around the border of the input.
- **Valid padding** ($P = 0$): output shrinks — spatial info at edges is lost faster.
- **Same padding** ($P = \lfloor K/2 \rfloor$): output size equals input size (when $S = 1$).

**Stride ($S$)**
- How many pixels the kernel jumps after each step.
- Larger stride → smaller output (downsampling without pooling).
- $S = 1$: densest sampling; $S = 2$: roughly halves spatial dimensions.

**Kernel Size ($K$)**
- Larger kernels capture more context per step but reduce output size more aggressively.
- Common choices: $3 \times 3$, $5 \times 5$, $7 \times 7$.

---

### Worked Examples

**Example 1 — Valid convolution (no padding, stride 1)**
$$
I = 28,\quad K = 3,\quad P = 0,\quad S = 1
$$
$$
\text{Output} = \left\lfloor \frac{28 - 3 + 0}{1} \right\rfloor + 1 = 26
$$

**Example 2 — Same convolution (output = input)**
$$
I = 28,\quad K = 3,\quad P = 1,\quad S = 1
$$
$$
\text{Output} = \left\lfloor \frac{28 - 3 + 2}{1} \right\rfloor + 1 = 28
$$

**Example 3 — Strided convolution (downsampling)**
$$
I = 28,\quad K = 3,\quad P = 0,\quad S = 2
$$
$$
\text{Output} = \left\lfloor \frac{28 - 3 + 0}{2} \right\rfloor + 1 = 13
$$

---

### Intuition

Think of the kernel as a sliding window. The number of valid positions it can occupy without going out of bounds (before padding) is the output size. Padding artificially extends the input so the window can visit more positions; stride skips positions.

The floor $\lfloor \cdot \rfloor$ handles cases where the kernel does not divide the input evenly — the last incomplete step is simply dropped (unless explicit padding is added).

---

### More Quick Examples

**Example 4 — No padding, stride 1 (Valid)**
$$I=7,\quad K=3,\quad P=0,\quad S=1 \implies O = (7-3+0)/1+1 = 5 \quad \text{(shrinks by 2)}$$

**Example 5 — Same padding, stride 1**
$$I=7,\quad K=3,\quad P=1,\quad S=1 \implies O = (7-3+2)/1+1 = 7 \quad \text{(same size)}$$

**Example 6 — No padding, stride 2**
$$I=8,\quad K=3,\quad P=0,\quad S=2 \implies O = \lfloor(8-3+0)/2\rfloor+1 = 3 \quad \text{(shrinks fast)}$$

**Example 7 — ResNet first layer (real-world)**
$$I=224,\quad K=7,\quad P=3,\quad S=2 \implies O = (224-7+6)/2+1 = 112$$

**Example 8 — Large kernel, no padding**
$$I=32,\quad K=5,\quad P=0,\quad S=1 \implies O = (32-5+0)/1+1 = 28 \quad \text{(shrinks by 4)}$$

### Summary Pattern

| Scenario | Effect |
|---|---|
| P=0, S=1 | Output shrinks by `K−1` |
| P=(K−1)/2, S=1 | Output stays same size |
| P=0, S=2 | Output roughly halves |
| P=same, S=2 | Output exactly halves |

---

## Q: Early layers learn generic features — how does this enable Transfer Learning?

### The Hierarchy of Learned Features

When a CNN is trained on a large dataset (e.g., ImageNet), its layers specialize progressively:

| Layer depth | What it detects |
|-------------|-----------------|
| Early (conv1–conv2) | Edges, color blobs, simple gradients |
| Middle (conv3–conv4) | Textures, corners, curves, simple shapes |
| Deep (conv5+) | Object parts, semantic concepts (eyes, wheels) |
| Final FC layers | Dataset-specific class embeddings |

This was empirically shown by Zeiler & Fergus (2014) via DeconvNet visualization — early-layer filters look nearly identical across CNNs trained on completely different tasks.

---

### Why Generic Features Transfer

Early features (edges, textures) are **task-agnostic**: virtually every vision problem benefits from detecting edges and textures. These features emerge from the statistics of natural images, not from the specific labels.

Because the early weights are already "good" for any visual task, a new model can:

1. **Freeze** those layers entirely → keep the generic feature extractor, train only the final classifier.
2. **Fine-tune** the whole network starting from pre-trained weights → converges faster, needs less data.
3. **Use as a fixed feature extractor** → pass images through the pretrained backbone, feed the output to an SVM or logistic regression.

---

### Practical Rule of Thumb

| Available data | Target task similarity to source | Strategy |
|----------------|----------------------------------|----------|
| Small | Similar | Freeze all, retrain only head |
| Small | Different | Freeze early layers, fine-tune middle+head |
| Large | Similar | Fine-tune entire network (small LR) |
| Large | Different | Train from scratch (or light fine-tune) |

**Why does this save data?** Pre-training acts as a structured prior. Instead of learning edge detectors from scratch (which needs millions of examples), the model starts with them for free.

---

## Q: How does data augmentation teach the model invariance to realistic transformations?

### What "Invariance" Means

A model is **invariant** to a transformation $T$ if:

$$
f(T(x)) = f(x)
$$

That is, applying the transformation to the input does not change the model's output (e.g., predicted class). In practice, we aim for approximate invariance — small, realistic transformations should not flip the predicted label.

---

### How Augmentation Teaches It

During training, augmentation applies random transformations to each image **while keeping the same label**. The model sees:

$$
(T_1(x),\, y),\quad (T_2(x),\, y),\quad (T_3(x),\, y), \quad \ldots
$$

The gradient updates teach the network that all these variants map to the same $y$. Over many epochs, the network's internal representations become *invariant* to those transformations — they activate similarly regardless of whether a flip or crop was applied.

---

### Augmentation → Invariance Mapping

| Augmentation | Invariance the model learns |
|--------------|-----------------------------|
| Horizontal flip | Mirror symmetry (useful for animals, cars; not for text/digits) |
| Random crop | Translation invariance |
| Color jitter / brightness | Photometric / illumination invariance |
| Rotation | Rotational invariance |
| Gaussian blur | Scale / frequency invariance |
| Cutout / random erasing | Occlusion invariance |

---

### Why "Realistic" Transformations Matter

Not every transformation is valid. For example:
- Rotating a digit **6** by 180° gives a **9** — wrong label. Aggressive rotation augmentation on digits would hurt accuracy.
- Vertical flip of a cat is unrealistic; the model wastes capacity learning it.

The rule: **augmentation should reflect the transformations the model will see at test/deployment time**, not arbitrary ones. Domain knowledge determines which augmentations are safe.

---

### Effect on Decision Boundary

Without augmentation, the decision boundary fits tightly around training samples — it is brittle to small input perturbations. With augmentation, the boundary becomes smoother and more spread out, because the model has been trained on a denser cloud of transformed examples. This directly connects to the next concept: variance reduction.

---

## Q: How does data augmentation reduce the variance of the learned model?

### Variance in the ML Sense

In the bias–variance framework:

$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$

**Variance** is how much the model's predictions change when it is retrained on a different (but same-distribution) training set. High variance = overfitting = the model memorized training samples rather than learning the underlying pattern.

---

### How Augmentation Reduces Variance

Data augmentation **artificially expands the effective training set size** by generating new examples on-the-fly from existing ones. A model trained on $N$ real images plus augmentation effectively sees $N \times k$ distinct inputs (where $k$ = number of augmentation variants).

Larger effective dataset → smoother, more stable learned function → lower variance.

More formally, each augmented version of an image adds a constraint to the optimization:

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\bigl(f(T_j(x_i)),\, y_i\bigr), \quad T_j \sim \mathcal{T}
$$

The network cannot simply memorize a specific pixel arrangement of $x_i$ because $T_j(x_i)$ is different each epoch. This forces the model to learn features that generalize across the transformation family $\mathcal{T}$.

---

### Bias–Variance Tradeoff Summary

| Condition | Bias | Variance | Result |
|-----------|------|----------|--------|
| No augmentation, small dataset | Low | **High** | Overfits |
| Augmentation, small dataset | Slightly higher | **Lower** | Better generalization |
| Augmentation, large dataset | Low | Low | Best generalization |

> Augmentation slightly increases bias (the model cannot perfectly fit individual training images) but substantially decreases variance — a net win.

---

### Concrete Example

A cat classifier trained on 1,000 images without augmentation may learn "cats have a horizontal orientation" because all training photos happen to be landscape. With horizontal-flip augmentation, both orientations are equally common, and the model generalizes to both — variance on unseen orientations is dramatically lower.

---

## Q: What is L2 regularization (weight decay), and how does it work?

### Modified Loss Function

L2 regularization adds a penalty term proportional to the **squared magnitude** of all weights:

$$
L_{\text{total}} = L_{\text{task}} + \lambda \sum_{i} w_i^2
$$

| Term | Role |
|------|------|
| $L_{\text{task}}$ | Original loss (cross-entropy, MSE, etc.) |
| $\lambda \sum_i w_i^2$ | Penalty — grows quadratically with weight magnitude |
| $\lambda$ | Regularization strength hyperparameter |

---

### Effect on the Gradient Update

Taking the derivative with respect to $w_i$:

$$
\frac{\partial L_{\text{total}}}{\partial w_i} = \frac{\partial L_{\text{task}}}{\partial w_i} + 2\lambda w_i
$$

The gradient descent update becomes:

$$
w_i \leftarrow w_i - \alpha \left( \frac{\partial L_{\text{task}}}{\partial w_i} + 2\lambda w_i \right)
$$

Rearranging:

$$
w_i \leftarrow w_i \underbrace{(1 - 2\alpha\lambda)}_{\text{decay factor}} - \alpha \frac{\partial L_{\text{task}}}{\partial w_i}
$$

The factor $(1 - 2\alpha\lambda)$ **shrinks** $w_i$ at every step, regardless of the gradient direction. This is why L2 regularization is also called **weight decay** — weights are literally decayed toward zero each update.

---

### Why Penalizing Large Weights Prevents Overfitting

- Large weights make the model **highly sensitive** to small input changes — a tiny pixel difference causes a large output change. This is exactly overfitting behavior.
- Penalizing $w_i^2$ biases the optimizer toward solutions with many small weights rather than a few extreme ones.
- **Occam's Razor** analogy: prefer the simpler model (smaller weight magnitudes) among those that fit the data equally well.

---

### Geometric Intuition

The L2 penalty corresponds to constraining weights to lie within a **sphere** in weight space:

$$
\sum_i w_i^2 \leq C
$$

The optimal unconstrained solution may lie outside the sphere. L2 pulls it toward the center (origin), landing on the sphere boundary where the loss contours and the constraint surface are tangent.

---

### L2 vs L1 Summary

| Property | L2 (Ridge) | L1 (Lasso) |
|----------|-----------|-----------|
| Penalty | $\lambda \sum w_i^2$ | $\lambda \sum \|w_i\|$ |
| Effect on weights | Shrinks all weights smoothly | Drives some weights to **exactly 0** |
| Produces sparsity? | No | Yes |
| Geometry | Circular (sphere) constraint | Diamond constraint |
| Gradient | Proportional to $w_i$ | Constant sign ($\pm\lambda$) |
| Best use | Default regularizer, dense models | Feature selection, sparse models |

---

## Q: What is L1 regularization, and why does it produce sparse weights?

### Modified Loss Function

L1 regularization adds a penalty proportional to the **absolute magnitude** of weights:

$$
L_{\text{total}} = L_{\text{task}} + \lambda \sum_{i} |w_i|
$$

### Gradient of the L1 Penalty

Unlike L2, the absolute value is not differentiable at zero. The sub-gradient is:

$$
\frac{\partial}{\partial w_i}(\lambda |w_i|) = \lambda \cdot \text{sign}(w_i) =
\begin{cases}
+\lambda & w_i > 0 \\
-\lambda & w_i < 0 \\
0 & w_i = 0
\end{cases}
$$

The update rule:

$$
w_i \leftarrow w_i - \alpha \left( \frac{\partial L_{\text{task}}}{\partial w_i} + \lambda \cdot \text{sign}(w_i) \right)
$$

---

### Why L1 Produces Exact Zeros (Sparsity)

- L2 decay: weight shrinkage is **proportional** to $w_i$ — as $w_i \to 0$, the penalty gradient also $\to 0$, so it never fully zeroes out.
- L1 decay: the penalty gradient is a **constant** $\pm\lambda$ regardless of how small $w_i$ is. If a weight is small and the task gradient is also small, the constant L1 push overshoots zero and the optimizer sets $w_i = 0$ exactly (via soft-thresholding).

This means L1 performs **automatic feature selection** — irrelevant features get their corresponding weights set to zero, and those features are effectively ignored.

---

### Geometric Diamond Intuition

The L1 constraint is:

$$
\sum_i |w_i| \leq C
$$

In 2D weight space, this constraint region is a **diamond** (rotated square). The corners of the diamond lie on the axes, meaning $w_1 = 0$ or $w_2 = 0$.

When the loss function's contour lines (ellipses) first touch the diamond, they are very likely to land on a **corner** — at an axis — producing a sparse solution. By contrast, the L2 sphere has no corners, so the touching point is rarely on an axis.

$$
\underbrace{\diamond}_{\text{L1 — corners on axes}} \quad \text{vs} \quad \underbrace{\bigcirc}_{\text{L2 — no corners}}
$$

---

### Sparsity = Automatic Feature Selection

In a linear model, if input $x_j$ is irrelevant, L1 drives $w_j \to 0$ exactly, effectively removing that feature from the model. This is valuable when:
- Input dimensionality is very high (many features, few relevant).
- You want an interpretable model — only the features with non-zero weights "matter."
- Memory/inference efficiency — sparse weight matrices can be compressed.

---

## Q: How does batch normalization provide implicit regularization?

### Batch Normalization — What It Does

For each mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ in a layer, BatchNorm normalizes the activations:

$$
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

where $\gamma$ and $\beta$ are learnable scale and shift parameters.

The primary motivation is **faster, more stable training** (reduces internal covariate shift). The regularization effect is a **side effect**.

---

### What "Implicit" Means

"Explicit" regularization (L1, L2, Dropout) is deliberately added with the goal of preventing overfitting.

"Implicit" regularization emerges as a byproduct of the learning process or architecture — BatchNorm was designed for training stability, not regularization, but it regularizes anyway.

---

### The Noise Mechanism

During training, $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are computed from a **random mini-batch** (typically 32–256 samples). These estimates are noisy — they differ from the true population mean $\mu$ and variance $\sigma^2$.

Each input $x_i$ is normalized using the statistics of its randomly sampled mini-batch. The normalization therefore changes slightly every time $x_i$ appears in a different batch.

$$
\hat{x}_i \text{ (batch 1)} \neq \hat{x}_i \text{ (batch 2)} \quad \text{even for the same } x_i
$$

This **stochastic perturbation** of activations acts like a noise injector — the model cannot rely on the exact activation values, so it must learn more robust, generalized representations.

---

### Why Noise = Regularization

- The network sees a slightly different version of each training example every epoch (because mini-batch statistics vary).
- This is conceptually similar to how **Dropout** injects noise by randomly zeroing activations.
- Both mechanisms prevent co-adaptation of neurons and force distributed representations.

| Property | Dropout | Batch Normalization |
|----------|---------|---------------------|
| Noise type | Binary mask on activations | Stochastic normalization |
| Applied at | Training only | Training (stochastic) / Test (deterministic) |
| Explicit regularizer? | Yes | No (implicit) |
| Regularization strength | Controlled by $p$ | Controlled by batch size |

---

### Train vs Test Discrepancy

This is a critical implementation detail:

**During training:** $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are computed from the mini-batch (stochastic, noisy).

**During inference:** Using a mini-batch mean would make predictions non-deterministic (different results for the same input depending on what other images are in the batch — unacceptable). Instead, BatchNorm uses **running averages** accumulated during training:

$$
\mu_{\text{running}} = \alpha \cdot \mu_{\text{running}} + (1 - \alpha) \cdot \mu_\mathcal{B}
$$

These running statistics are fixed at test time, so the normalization is deterministic.

> **Implication:** BatchNorm layers must be set to `eval()` mode at inference time (in PyTorch: `model.eval()`), or predictions will be incorrect.

---

### Practical Implication

Because BatchNorm already provides implicit regularization through its noise mechanism, models using BatchNorm typically need **less Dropout** than models without it. Over-regularizing with both strong Dropout and BatchNorm can hurt performance (too much noise during training → underfitting).

Common practice:
- CNN classification: BatchNorm in every conv block, light or no Dropout.
- FC heads: sometimes add Dropout ($p = 0.3$–$0.5$) after BatchNorm.

---

---

## Q: Multi-head attention — use multiple attention heads in parallel, each focusing on different aspects

### Single-Head Attention Recap

Standard (single-head) scaled dot-product attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Each token attends to every other token, producing a weighted sum of values. The problem: **one attention head can only focus on one type of relationship at a time** — it can't simultaneously focus on syntactic structure *and* semantic similarity *and* positional proximity.

---

### Multi-Head Attention — The Idea

Instead of running one attention function with full dimensionality, you run **$h$ attention heads in parallel**, each with its own learned projection:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

| Symbol | Shape | Role |
|--------|-------|------|
| $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ | Projects queries into head $i$'s subspace |
| $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ | Projects keys into head $i$'s subspace |
| $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ | Projects values into head $i$'s subspace |
| $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ | Projects concatenated heads back |

Typically $d_k = d_v = d_{\text{model}} / h$, so total compute ≈ single-head cost.

---

### What "Different Aspects" Actually Means

Each head learns different projection matrices ($W_i^Q, W_i^K, W_i^V$), so it attends to a **different subspace** of the representation. In practice, heads empirically specialize:

| Head type (observed) | What it focuses on |
|----------------------|--------------------|
| Positional heads | Attend to tokens at fixed relative distances (e.g., previous token) |
| Syntactic heads | Attend along dependency-parse relationships (subject → verb) |
| Semantic heads | Attend to semantically similar or co-referent tokens |
| Rare-word heads | Attend broadly when uncertain, sharply on known words |

No head is *programmed* to do this — it emerges from training. The model discovers which decomposition of attention is most useful.

---

### Why Multiple Heads Outperform One Fat Head

A single head with $d_{\text{model}}$ dimensions is forced to average all these relationships into a single attention distribution (softmax produces one scalar weight per token pair). That's a bottleneck.

With $h$ heads, each head has its **own** softmax distribution — head 1 can strongly attend to position $j$ while head 2 strongly attends to position $k$, simultaneously. The model gets $h$ independent "views" of the sequence.

Think of it like an ensemble of attention mechanisms all running in parallel and then combined.

---

### In Vision (ViT / Vision Transformers)

In Vision Transformers, the input is split into patches (tokens). Multi-head attention over patches allows:
- Some heads to attend to **local texture patches** (nearby patches)
- Some heads to attend to **global structure** (distant patches — e.g., matching left eye to right eye)
- Some heads to track **class-relevant regions** regardless of position

This is why ViTs can capture both local and global context simultaneously, something CNNs achieve only through stacking many layers.

---

### Complexity Note

$$
\text{Compute per head} = O(n^2 \cdot d_k), \quad \text{Total} = O(n^2 \cdot d_{\text{model}})
$$

Multi-head doesn't increase complexity over single-head (because $d_k = d_{\text{model}}/h$), but it substantially increases **representational capacity** by learning diverse attention patterns.

---

---

## Q: Why ViT Works — Inductive biases not always needed: with enough data, the network can learn useful spatial structures without hard-coded biases

### What is an Inductive Bias?

An **inductive bias** is an assumption baked into the model architecture that restricts the hypothesis space — it tells the model *how* to generalize before seeing any data.

CNNs have two strong inductive biases hard-coded into their architecture:

| Inductive Bias | What it assumes | How it's implemented |
|----------------|-----------------|----------------------|
| **Translation equivariance** | A cat in the top-left and a cat in the bottom-right are the same | Shared (tied) convolution weights across all spatial positions |
| **Locality** | Nearby pixels are more related than distant ones | Small kernel windows (3×3, 5×5) — each neuron sees only a local patch |

These biases are enormously helpful when data is scarce — they reduce the number of parameters the model has to learn and prevent it from wasting capacity on irrelevant spatial relationships.

---

### What ViT Gives Up

A Vision Transformer treats an image as a flat sequence of patches and applies **self-attention globally** from layer one. It has:

- **No translation equivariance** — the same patch in position 5 vs position 50 is treated differently (it has different positional embeddings).
- **No locality bias** — every patch attends to every other patch with equal "privilege" from the start; there's no architectural preference for nearby patches.

In other words, ViT is a much more **general function approximator** for images. It makes far fewer assumptions about spatial structure.

---

### Why That's a Problem with Small Data

Because ViT makes no assumptions, it has to **learn everything from scratch**:
- Learn that nearby patches tend to be correlated.
- Learn that a feature appearing in one location should generalize to other locations.
- Learn which spatial relationships matter for the task.

With limited data (e.g., ImageNet-1k, ~1.2M images), ViT doesn't have enough signal to discover all of this reliably. CNNs win here because their hard-coded biases give them a head start — they never need to "discover" locality, it's already built in.

Dosovitskiy et al. (2020, "An Image is Worth 16×16 Words") explicitly showed this: ViT trained on ImageNet-1k alone **underperforms** ResNets of comparable size.

---

### Why Enough Data Removes the Need for Inductive Bias

With a massive dataset like **ImageNet-21k (~14M images)** or **JFT-300M (~300M images)**, the model sees enough diverse examples that it can *learn* the spatial regularities that CNNs have hard-coded:

- It sees cats in every position, so it learns position-invariant representations empirically.
- It sees enough local co-occurrences that attention heads naturally specialize to attend locally.
- The sheer volume of gradient signal is enough to shape the weights into something functionally similar to what CNN inductive biases give for free.

Formally: inductive bias trades **data** for **assumptions**. The more data you have, the less valuable a restrictive assumption is — and the more it can actually *hurt* by preventing the model from learning something better.

$$
\text{Effective learning} = \text{Data} + \text{Inductive Bias}
$$

- Low data → high bias needed → CNNs win.
- High data → bias becomes a constraint, not a help → flexible models (ViT) win.

---

### The Practical Payoff

Once pre-trained on sufficient data, ViT's general representations **transfer better** than CNNs to diverse downstream tasks. Because it made fewer assumptions during pre-training, its features are less "CNN-shaped" and more universally useful. This is why ViT and its descendants (DeiT, Swin, DINO) dominate modern CV benchmarks when pre-training data is abundant.

---

---

## Q: CNNs win with small data — hard-coded biases give them a head start, they never need to "discover" locality

### The Core Idea

When you train any neural network, every useful property it needs must come from **one of two places**:
1. **Learned from data** — the optimizer discovers it through gradient descent.
2. **Built into the architecture** — it's true by construction, no data needed.

CNNs get locality and translation equivariance from category 2. ViT must get them from category 1. That difference is everything when data is scarce.

---

### What "Discovering Locality" Would Cost ViT

Imagine ViT starts with random weights and only 10,000 training images. It needs to figure out, purely from data, that:

- Pixel at position $(i, j)$ is more informative about pixel $(i, j+1)$ than about pixel $(i, j+100)$.
- The concept "edge" is a local relationship between neighboring pixels, not a global one.
- Texture and shape signals come from compact spatial neighborhoods.

To *learn* this, ViT's attention weights for nearby patches need to consistently come out higher than for distant patches — across millions of weight updates, across thousands of diverse images. With 10,000 images, the gradient signal is too weak and too noisy to reliably push the weights in this direction. The model may partially learn it, partially not — producing unstable, poorly generalized features.

---

### What CNN Gets for Free

A CNN with a $3 \times 3$ kernel **physically cannot** look at non-adjacent pixels in a single layer. The architecture enforces:

$$
\text{output}(i, j) = f\bigl(\text{input}[i-1:i+2,\; j-1:j+2]\bigr)
$$

There are no weights connecting $(i, j)$ to $(i, j+50)$ — the connection literally doesn't exist. The model doesn't need to *discover* that nearby pixels matter; the architecture makes it impossible to do anything else.

This means **every gradient step** is already spent learning *which* local patterns are useful (edges, corners, textures), not *whether* to look locally at all. Enormous efficiency gain.

---

### Translation Equivariance: The Same Saving

CNNs use **weight sharing** — the same $3 \times 3$ filter is applied at every spatial position. This means:

- A filter that detects a vertical edge at position $(10, 10)$ automatically detects it at $(50, 80)$ too — same weights, different position.
- The model never needs to learn "vertical edge detector for the top-left corner" separately from "vertical edge detector for the bottom-right corner."

For a $224 \times 224$ image with a $3 \times 3$ filter, weight sharing reduces the parameters for one filter from $224 \times 224 \times 9 = 451{,}584$ to just $9$. That's a **50,000× parameter reduction** for one filter — all of it funded by the locality + equivariance assumption.

ViT has no weight sharing across positions. Its attention patterns at position 5 and position 50 are completely independent — each must be learned separately.

---

### The Data Budget Analogy

Think of training data as a **budget** you spend to teach the model things:

| Thing to learn | CNN cost | ViT cost |
|----------------|----------|----------|
| "Look at local neighborhoods" | $0 — built in | High — must be discovered |
| "Same filter works everywhere" | $0 — built in | High — no weight sharing |
| "Which edges matter for this class" | Full budget | Partial budget (rest spent on basics) |

With a small dataset, CNN spends its entire budget on the *interesting* task-specific question. ViT wastes budget re-discovering spatial basics that CNN gets for free.

With a massive dataset, the budget is so large that ViT can afford to pay the "basics" cost and still have plenty left for the interesting stuff — and then its lack of restrictive assumptions becomes an advantage.

---

### One-Line Summary

> CNN inductive biases are a **compressed prior** about the statistics of natural images. They're almost always true (images are locally structured), so encoding them architecturally is nearly free lunch — until you have so much data that you can afford to learn them, and the prior starts constraining you.

---

---

## Q: Hybrid Models — Combine Convolution and Attention

### The Motivation

Pure CNNs and pure ViTs each have complementary strengths and weaknesses:

| | CNN | ViT |
|--|-----|-----|
| Local feature extraction | Excellent (built-in) | Weak (must learn) |
| Global context modeling | Weak (needs many layers) | Excellent (attention is O(1) hops) |
| Data efficiency | High | Low (needs large pretraining) |
| Computational cost | Lower | Higher ($O(n^2)$ attention) |
| Inductive biases | Strong (locality, equivariance) | Minimal |

The natural question: **what if you use each where it's strongest?**

---

### The General Hybrid Recipe

$$
\underbrace{\text{Conv layers}}_{\text{local features, cheap}} \rightarrow \underbrace{\text{Attention layers}}_{\text{global context, expressive}}
$$

Early stages use convolutions to extract rich local features (edges, textures, object parts) at high spatial resolution efficiently. Later stages use attention to reason about relationships between those features globally — but now the sequence is much shorter (feature map, not raw pixels), making attention affordable.

---

### Architecture Patterns

**Pattern 1 — Conv stem + Transformer body (e.g., ViT with conv patch embedding)**

Instead of splitting the image into flat non-overlapping patches (raw ViT), a small CNN processes the image first, then its output feature map is fed as the token sequence to the Transformer.

- Conv stem learns overlapping, multi-scale local features.
- Transformer body attends over those features globally.
- Result: better data efficiency than pure ViT, better global reasoning than pure CNN.

**Pattern 2 — Conv blocks + Self-attention blocks interleaved (e.g., BoTNet, CoAtNet)**

Standard ResNet blocks are used for most of the network. The final few stages replace $3 \times 3$ convolutions with **multi-head self-attention**:

$$
\text{Conv} \to \text{Conv} \to \text{Conv} \to \underbrace{\text{Attention} \to \text{Attention}}_{\text{last stages only}}
$$

By the last stages, the feature map is spatially small (e.g., $7 \times 7 = 49$ tokens), so the $O(n^2)$ attention cost is negligible. These stages need global context most — deciding if a torso and a head belong to the same person, for example.

**Pattern 3 — Swin Transformer (local attention → global via shifting)**

Uses convolution-like **local windows** for attention (each token only attends to its $7 \times 7$ neighborhood), then shifts the windows each layer so information propagates globally over depth. Gets the efficiency of CNNs with the flexibility of attention.

---

### Why This Works Better Than Either Alone

1. **Low-level features are local by nature** — a $3 \times 3$ conv is the right tool. Spending attention on raw pixels is wasteful.
2. **High-level reasoning is global** — "is this the same object across a scene?" requires attention across the full feature map.
3. **Attention on conv features is cheaper** — after several conv+pool stages, spatial resolution shrinks from $224 \times 224$ to $14 \times 14$, reducing attention cost from $50{,}176^2$ to $196^2$ — a **65,000× reduction**.
4. **Better transfer** — conv layers provide strong low-level priors; attention layers provide flexible high-level reasoning. Both contribute to richer pretrained representations.

---

### Notable Hybrid Models

| Model | Strategy | Key Insight |
|-------|----------|-------------|
| **BoTNet** | ResNet + self-attention in last 3 blocks | Minimal change to ResNet, big gain on detection |
| **CoAtNet** | Conv early stages + Transformer late stages | Systematic study of where each works best |
| **Swin Transformer** | Local windowed attention + shift | Hierarchical like CNN, attention everywhere |
| **CvT** | Conv projections inside Transformer | Adds locality to the Q/K/V projections |
| **ConvNeXt** | Pure CNN modernized with Transformer design choices | Shows CNN can match ViT with right design |

---

### One-Line Summary

> Hybrid models let convolutions do what they do cheaply and perfectly (local features), then hand off to attention for what it does uniquely well (global reasoning) — getting the best of both worlds at a fraction of the cost of pure attention.

---

---

## Q: Why is spending attention on raw pixels wasteful?

### What Attention Actually Does

Self-attention computes a relationship score between **every pair** of tokens in the sequence:

$$
\text{Attention cost} = O(n^2 \cdot d)
$$

where $n$ = number of tokens, $d$ = feature dimension. The $n^2$ term is the killer — it means cost grows quadratically with sequence length.

---

### The Raw Pixel Problem

A $224 \times 224$ image has $224 \times 224 = 50{,}176$ pixels. If you feed raw pixels as tokens:

$$
n = 50{,}176 \quad \Rightarrow \quad n^2 = 2{,}517{,}630{,}976 \approx 2.5 \text{ billion pairs}
$$

That's 2.5 billion attention scores to compute **per layer, per image**. Computationally brutal.

But the bigger problem isn't just cost — it's **what those attention scores would mean**.

---

### Raw Pixels Carry Almost No Semantic Information

Consider pixel $(100, 100)$ in an image of a dog. Its raw RGB value might be `(180, 160, 140)` — a brownish color. Now ask: what should this pixel *attend to*?

- Is it part of the dog's fur? The background? A shadow?
- Is it related to the pixel 3 positions away, or the one 200 positions away?

**A raw pixel value alone cannot answer any of these questions.** The pixel has no context about what object it belongs to, what edge it's near, or what texture it's part of. The attention mechanism would be trying to find meaningful relationships between meaningless numbers.

Attention is designed to relate **semantic tokens** — words in a sentence, object patches in an image — not raw intensity values.

---

### What Needs to Happen First

Before attention can be useful, the pixels need to be transformed into something that carries **local semantic meaning**. That's exactly what convolutions do:

$$
\underbrace{\text{raw pixel } (180, 160, 140)}_{\text{no meaning}} \xrightarrow{\text{conv layers}} \underbrace{\text{feature vector: "fur texture, warm tone, smooth gradient"}}_{\text{meaningful}}
$$

After a few conv layers, each spatial location's feature vector encodes what kind of visual structure exists there — edges, textures, parts. *Now* attention can ask a meaningful question: "does this fur-texture region relate to that ear-shape region?"

---

### The Resolution Shrinkage Bonus

Conv layers also downsample spatially (via stride or pooling):

| Stage | Spatial size | Tokens for attention |
|-------|-------------|----------------------|
| Raw pixels | $224 \times 224$ | $50{,}176$ |
| After 1 conv block (stride 2) | $112 \times 112$ | $12{,}544$ |
| After 3 conv blocks (stride 8) | $28 \times 28$ | $784$ |
| After 4 conv blocks (stride 16) | $14 \times 14$ | $196$ |

Going from $50{,}176$ to $196$ tokens reduces attention cost by:

$$
\frac{50{,}176^2}{196^2} \approx 65{,}000\times
$$

And those 196 tokens are now **rich feature vectors**, not raw pixel values. You get cheaper attention *and* more meaningful attention simultaneously.

---

### Analogy

Asking attention to work on raw pixels is like asking a literature professor to find thematic connections between individual letters of a novel instead of between sentences and paragraphs. The letters are the raw data, but the meaning lives at a higher level of abstraction. You need to build up structure (words → sentences → paragraphs) before cross-referencing makes sense.

Convolutions build that structure. Then attention cross-references it.

---

*End of notes — continued in next session.*
