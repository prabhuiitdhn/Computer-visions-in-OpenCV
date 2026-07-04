# Diffusion Models — Interview Prep: Beginner to Expert

> Session date: 2026-07-02  
> Goal: Preparing for Senior Researcher role interview in Machine Vision / Generative AI

**References:**
- [Diffusion Models — HuggingFace Blog (Esmail AGumaan)](https://huggingface.co/blog/Esmail-AGumaan/diffusion-models)

---

## Level 1: The "What" — Conceptual Foundation

### What is a Diffusion Model?

A **Diffusion Model** is a class of **generative models** that learn to generate data (images, audio, video) by learning to **reverse a noise corruption process**.

The core idea is borrowed from **non-equilibrium thermodynamics**:

> If you gradually add Gaussian noise to an image until it becomes pure noise, can you learn to reverse that process?

The answer is **yes** — and that's the entire premise.

---

### Two Processes

| Process | Direction | Learnable? |
|---|---|---|
| **Forward Process** $q$ | Data → Noise | No (fixed) |
| **Reverse Process** $p_\theta$ | Noise → Data | Yes (neural net) |

---

### The Forward Process (Fixed)

Given a clean image $x_0$, we add Gaussian noise over $T$ timesteps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t \mathbf{I})$$

Where $\beta_t$ is a **noise schedule** (small values, e.g. 0.0001 → 0.02).

A key trick (reparameterization) lets you jump to any timestep directly:

$$q(x_t \mid x_0) = \mathcal{N}(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t)\mathbf{I})$$

Where:

$$\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$$

So at $t = T$: $x_T \approx \mathcal{N}(0, \mathbf{I})$ — pure noise.

---

### The Reverse Process (Learned)

The model learns:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t))$$

In practice (DDPM — Ho et al. 2020), the network **predicts the noise** $\epsilon$ that was added, not the mean directly.

The training objective simplifies beautifully to:

$$\mathcal{L} = \mathbb{E}_{x_0,\, \epsilon,\, t} \left[ \lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2 \right]$$

**That's it** — just MSE between the actual noise and the predicted noise.

---

### Why Diffusion Models? (vs GANs, VAEs)

| Model | Strength | Weakness |
|---|---|---|
| **GAN** | Sharp images, fast | Training instability, mode collapse |
| **VAE** | Stable training, latent space | Blurry outputs |
| **Diffusion** | High quality, stable, diverse | Slow sampling ($T = 1000$ steps) |

Diffusion models **dominate** image generation benchmarks (FID scores) because they avoid adversarial training instability while producing sharper outputs than VAEs.

---

## Topics to Explore Next

1. **Architecture** — What neural network sits inside? (U-Net, attention, transformers)
2. **Sampling speed** — DDIM, consistency models (solving the slow inference problem)
3. **Conditioning** — How text-to-image works (CLIP, classifier-free guidance)
4. **Latent Diffusion** — How Stable Diffusion moves to latent space for efficiency
5. **Score Matching** — The theoretical connection to score-based generative models
6. **Vision applications** — Inpainting, super-resolution, medical imaging, depth estimation

---

## Level 2: Architecture — What Neural Network Sits Inside?

The neural network inside a diffusion model must solve one problem:

> Given a noisy image $x_t$ and a timestep $t$, predict the noise $\epsilon$ that was added.

This requires understanding **both local texture** (what pixels look like) and **global structure** (what the image means). That's why the architecture evolved the way it did.

---

### The Backbone: U-Net

The original DDPM (Ho et al. 2020) uses a **U-Net** — originally designed for medical image segmentation.

```
Input x_t  →  [Encoder]  →  Bottleneck  →  [Decoder]  →  Predicted ε
                  ↓               ↑
              Skip Connections (residual paths)
```

**Why U-Net?**

- The **encoder** downsamples: captures high-level semantics (what is in the image)
- The **decoder** upsamples: reconstructs spatial detail (where things are)
- **Skip connections** preserve fine-grained spatial information that downsampling would destroy
- Output is the **same resolution** as input — perfect for predicting per-pixel noise

---

### Timestep Conditioning — Sinusoidal Embeddings

The network must know *which* timestep it's denoising at. This is injected via **sinusoidal positional embeddings** (same idea as in Transformers):

$$\text{emb}(t)_{2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{emb}(t)_{2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

This embedding is projected and **added to every residual block** via scale-shift (AdaGN):

$$\text{out} = \gamma(t) \cdot \text{GroupNorm}(x) + \delta(t)$$

---

### Adding Attention — The Critical Upgrade

Pure convolutions are local. But generating coherent images requires **long-range dependencies** (e.g., two eyes must be symmetric). The solution: inject **self-attention** layers at low-resolution feature maps.

**Where attention is placed:**

```
Resolution 64×64  → Conv blocks
Resolution 32×32  → Conv blocks
Resolution 16×16  → Conv + Self-Attention   ← long-range
Resolution  8×8   → Conv + Self-Attention   ← global context
```

Attention at full resolution is $O(HW)^2$ — too expensive. At 8×8 it's cheap.

**Self-attention inside the U-Net:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q, K, V$ are linear projections of the spatial feature map, reshaped as a sequence of tokens.

---

### The Full Residual Block (DDPM / Improved DDPM)

```
x  →  GroupNorm → SiLU → Conv
                              ↓
t_emb → Linear → [scale, shift] → applied via AdaGN
                              ↓
       GroupNorm → SiLU → Dropout → Conv
                              ↓
       + residual connection (1×1 conv if channel mismatch)
```

---

### Evolution: U-Net → DiT (Diffusion Transformer)

In 2023, **Peebles & Xie** proposed replacing the U-Net with a pure **Vision Transformer** backbone — the **DiT (Diffusion Transformer)**.

| | U-Net (DDPM era) | DiT (2023+) |
|---|---|---|
| Backbone | CNN + Attention | Pure Transformer |
| Spatial processing | Convolutions | Patch tokenization |
| Timestep inject | AdaGN in residual blocks | AdaLN-Zero in every block |
| Scaling | Harder to scale | Scales like LLMs (compute = quality) |
| Used in | DDPM, DALL·E 2 | Stable Diffusion 3, Sora |

**DiT key idea:** Treat the noisy image as a sequence of patches (like ViT), then apply transformer blocks with adaptive layer norm conditioned on $t$.

$$\text{AdaLN-Zero: } y = \gamma(c) \cdot \text{LayerNorm}(x) + \delta(c)$$

where $c$ encodes both timestep and class/text condition. The "Zero" means $\gamma$ is initialized to 0 — the block starts as an identity, which stabilizes training.

---

### Summary: Architecture Lineage

```
DDPM (2020)          → U-Net, ResBlocks, no attention
Improved DDPM (2021) → U-Net + Self-Attention at low-res, learned noise schedule
LDM / Stable Diff    → U-Net in latent space + Cross-Attention for text conditioning
DiT (2023)           → Pure Transformer, patch tokens, AdaLN-Zero
Sora / SD3 (2024)    → DiT at scale (3B+ params), video/multi-modal
```

> **Key interview insight:** The shift from U-Net → DiT mirrors what happened in vision generally (CNN → ViT). The transformer's ability to scale predictably with compute is why the frontier models all use DiT now.

---

## Level 3: Sampling Speed — DDIM, PNDM, Consistency Models

### The Core Problem

DDPM requires **1000 sequential denoising steps** at inference. Each step = one full U-Net forward pass. On a GPU, that's **~30–60 seconds per image**. Completely impractical for production.

The goal: same quality, far fewer steps.

---

### Why Can't You Just Skip Steps in DDPM?

DDPM's reverse process is a **Markov chain** — each step depends on the previous:

$$p_\theta(x_{t-1} \mid x_t)$$

Skipping steps (e.g., $x_{1000} \to x_{500}$) breaks the Markov assumption. The noise distribution at $x_{500}$ doesn't match what the model was trained on for a 500-step jump. Quality collapses.

The fix: **re-derive a non-Markovian process** that allows large jumps.

---

### DDIM — Denoising Diffusion Implicit Models (Song et al. 2020)

**Key insight:** The DDPM training objective only depends on the marginals $q(x_t \mid x_0)$, not the joint. You can define a *different*, non-Markovian forward process with the **same marginals** but a deterministic reverse.

The DDIM update rule (deterministic, $\sigma=0$):

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } x_0} + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta(x_t, t)$$

Breaking this down:
1. **Predict $x_0$** from current $x_t$ using the noise prediction
2. **Re-noise** that predicted $x_0$ to level $t-1$

This is valid for **any subsequence** of timesteps $\{t_1, t_2, \ldots, t_S\} \subset \{1, \ldots, T\}$.

| | DDPM | DDIM |
|---|---|---|
| Steps needed | 1000 | **20–50** |
| Stochastic? | Yes | No (deterministic) |
| Same model weights? | — | **Yes** (no retraining) |
| Latent consistency | No | **Yes** (same noise → same image) |

DDIM also gives you a proper **latent space** — the same initial noise always produces the same image, enabling interpolation.

---

### PNDM — Pseudo Numerical Methods (Liu et al. 2022)

DDIM treats the reverse process as a first-order ODE solver (Euler method). **PNDM** treats it as a higher-order numerical ODE problem.

The diffusion reverse ODE:

$$\frac{dx}{dt} = f(x, t) - \frac{g(t)^2}{2}\, \nabla_x \log p_t(x)$$

Higher-order solvers (like Runge-Kutta) take **larger steps with less error**:

| Solver | Order | Steps for good quality |
|---|---|---|
| DDIM | 1st (Euler) | 50 |
| PNDM | 4th (pseudo linear multistep) | **20** |
| DPM-Solver++ | 2nd–3rd | **10–15** |

**PNDM key idea:** Use a linear multistep method — incorporate the gradient history from previous steps to better estimate the current step direction. No retraining needed.

> This is why Stable Diffusion ships with PNDM as its default scheduler.

---

### Consistency Models (Song et al. 2023) — The Paradigm Shift

DDIM/PNDM still need 10–50 steps. Consistency Models target **1–4 steps** — a fundamentally different approach.

**Core idea:** Train a function $f_\theta(x_t, t)$ that maps **any point on the diffusion trajectory directly to $x_0$**.

$$f_\theta(x_t, t) = x_0 \quad \text{for all } t$$

The **consistency property**:

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \text{for all } t, t' \text{ on the same trajectory}$$

All noisy versions of the same image must map to the same clean image.

**Two training modes:**

| Mode | How | Needs pretrained diffusion model? |
|---|---|---|
| **Consistency Distillation (CD)** | Distill from a pretrained DDPM | Yes |
| **Consistency Training (CT)** | Train from scratch with consistency loss | No |

**Consistency loss (distillation):**

$$\mathcal{L}_{CD} = \mathbb{E}\left[ d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\ f_{\theta^-}(x_{t_n}^{\Phi}, t_n)\right) \right]$$

Where:
- $\theta^-$ is an EMA (exponential moving average) of $\theta$ — the "target network" (stabilizes training, like in DQN)
- $x_{t_n}^{\Phi}$ is one ODE solver step from $x_{t_{n+1}}$
- $d(\cdot, \cdot)$ is a perceptual distance (LPIPS works better than MSE here)

**Multistep refinement:** Start with 1-step, then iteratively add noise and denoise for more steps if needed:

```
1-step:  noise → f_θ → x̂₀                        (fast, less sharp)
2-step:  noise → f_θ → add noise → f_θ → x̂₀      (better)
4-step:  ...                                        (near-DDPM quality)
```

---

### The Full Landscape of Samplers

```
Speed  ◄─────────────────────────────────────► Quality
  │                                               │
  │  Consistency (1-4 steps)                      │
  │  LCM / Turbo models (1-8 steps)               │
  │  DPM-Solver++ (10-15 steps)                   │
  │  PNDM (20 steps)                              │
  │  DDIM (50 steps)                              │
  │  DDPM (1000 steps)                            │
  │                                               │
```

**LCM (Latent Consistency Models, 2023):** Consistency distillation applied in **latent space** (like Stable Diffusion). Enables 4-step high-quality generation with SD architecture — no architecture change.

---

> **Key interview insight:** The sampling speed problem is essentially: **how do you solve a stochastic differential equation (SDE) or ODE with fewer function evaluations without losing accuracy?**
>
> - DDIM → 1st-order ODE solver
> - PNDM/DPM-Solver++ → higher-order ODE solvers
> - Consistency Models → learn the solution map directly, bypassing iterative solving entirely
>
> This framing — diffusion as a **continuous-time SDE/ODE** — is the foundation of **Score-Based Generative Models** (Topic 5), which unifies all of these methods theoretically.

---

## Level 4: Conditioning — Cross-Attention, CLIP, Classifier-Free Guidance

### How Text Controls Image Generation

Without conditioning, diffusion models generate **random images**. Conditioning is what lets you say *"a cat sitting on a red sofa"* and get exactly that.

There are three interlocking pieces: **CLIP** (text understanding), **Cross-Attention** (injecting text into the U-Net), and **Classifier-Free Guidance** (making the model *listen* to the text strongly).

---

### Piece 1: CLIP — Bridging Text and Images

**CLIP (Contrastive Language-Image Pretraining, OpenAI 2021)** is trained on 400M image-text pairs with a contrastive objective:

$$\mathcal{L}_{CLIP} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)}$$

Where $\text{sim}(I, T) = \frac{f_I(I) \cdot f_T(T)}{\|f_I(I)\|\,\|f_T(T)\|}$ is cosine similarity between image and text embeddings, and $\tau$ is a learned temperature.

**Result:** A shared embedding space where *"a dog"* and a photo of a dog land close together.

In Stable Diffusion, only the **text encoder** of CLIP is used — it converts a text prompt into a sequence of token embeddings:

$$\tau_\theta(\text{"a cat on a sofa"}) \to \mathbf{c} \in \mathbb{R}^{77 \times 768}$$

77 tokens (CLIP's max), 768-dim each. This is the conditioning signal $\mathbf{c}$.

---

### Piece 2: Cross-Attention — Injecting Text into the U-Net

The U-Net's spatial features $\phi(x_t) \in \mathbb{R}^{h \times w \times d}$ need to attend to the text tokens $\mathbf{c} \in \mathbb{R}^{77 \times 768}$.

**Cross-attention** (Rombach et al. 2022 — Stable Diffusion):

$$Q = W_Q \cdot \phi(x_t), \quad K = W_K \cdot \mathbf{c}, \quad V = W_V \cdot \mathbf{c}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Queries** come from the **image features** — "what does each spatial location need to know?"
- **Keys and Values** come from the **text tokens** — "what information does each word carry?"

Every spatial location in the feature map can attend to every word. This is how *"red"* influences *"sofa"* pixels specifically.

**Where cross-attention is added:**

```
U-Net Decoder Block:
  ResBlock(x_t, t_emb)        ← timestep conditioning (AdaGN)
       ↓
  Self-Attention(x)            ← spatial coherence
       ↓
  Cross-Attention(x, c)        ← text conditioning  ← HERE
       ↓
  output
```

Only the cross-attention weights $W_Q, W_K, W_V$ are trained during text-conditioned fine-tuning. This is why **LoRA** (Low-Rank Adaptation) works so well for SD fine-tuning — it adapts only these projection matrices.

---

### Piece 3: Classifier-Free Guidance (CFG) — The Strength Dial

Even with cross-attention, the model can "ignore" the text and just generate plausible-looking images. CFG forces the model to take the conditioning **seriously**.

**Setup:** During training, randomly drop the text condition with probability $p$ (typically 10–20%), replacing $\mathbf{c}$ with a null token $\emptyset$. The model learns both:

- $\epsilon_\theta(x_t, t, \mathbf{c})$ — conditioned noise prediction
- $\epsilon_\theta(x_t, t, \emptyset)$ — unconditional noise prediction

**At inference**, the final noise prediction is:

$$\tilde{\epsilon}_\theta(x_t, t, \mathbf{c}) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left[\epsilon_\theta(x_t, t, \mathbf{c}) - \epsilon_\theta(x_t, t, \emptyset)\right]$$

Where $w$ is the **guidance scale** (the CFG slider in UIs, typically 7–12).

**Geometric intuition:**

```
unconditional prediction
        ●
         \
          \  ← guidance pushes further in this direction
           \
            ● conditioned prediction
             \
              \  w = 7.5 → final prediction lands here
               ●
```

| Guidance Scale $w$ | Effect |
|---|---|
| $w = 1$ | No guidance (ignores text) |
| $w = 7$–$12$ | Standard — text-faithful, good quality |
| $w > 15$ | Over-saturated, artifact-prone |
| $w = 0$ | Fully unconditional |

**Cost:** CFG requires **two forward passes** per step (conditioned + unconditioned). Techniques like **CFG distillation** (used in SDXL-Turbo) bake guidance into the model weights to avoid this.

---

### How It All Connects: Full Text-to-Image Pipeline

```
Prompt: "a cat sitting on a red sofa"
        ↓
   CLIP Text Encoder
        ↓
   c ∈ ℝ^{77×768}  (text tokens)
        ↓
   Sample xT ~ N(0, I)  (random noise)
        ↓
   For t = T → 1:
     ε_cond   = U-Net(xT, t, c)      ← cross-attention to text
     ε_uncond = U-Net(xT, t, ∅)
     ε_guided = ε_uncond + w·(ε_cond - ε_uncond)   ← CFG
     x_{t-1}  = DDIM_step(xt, ε_guided)
        ↓
   x0  →  VAE Decoder  →  Final Image
```

---

### Advanced: Other Conditioning Modalities

The same cross-attention mechanism generalises beyond text:

| Condition | Encoder used | Application |
|---|---|---|
| Text | CLIP / T5 | Text-to-image (SD, DALL·E) |
| Image (reference) | CLIP image encoder | Image variation, style transfer |
| Depth map | Small CNN | ControlNet depth |
| Pose keypoints | CNN | ControlNet pose |
| Segmentation mask | CNN | ControlNet seg |
| Class label | Embedding lookup | Class-conditional DiT |

**ControlNet (Zhang et al. 2023):** Freezes the original U-Net and adds a **trainable copy of the encoder** that ingests spatial conditioning (depth, pose, edges). The outputs are added to the frozen decoder via zero-convolutions — enabling precise spatial control without degrading the original model.

---

> **Key interview insight:** CFG is not a separate classifier — it's a **score extrapolation** trick. You're moving the score estimate *further* in the direction the text pulls it, amplifying the signal. Formally it relates to the score of a sharpened distribution $p(x \mid c)^w / Z$. This is why high CFG produces oversaturated images — you're sampling from an increasingly peaked distribution.

---

## Level 5: Latent Diffusion — Why Stable Diffusion Works in Latent Space

### The Fundamental Problem with Pixel-Space Diffusion

DDPM and early diffusion models operate **directly on pixels**. For a 512×512 RGB image:

$$\text{Pixel space dimension} = 512 \times 512 \times 3 = 786{,}432$$

Every forward/reverse step processes ~786K values. The U-Net must run 1000 times on this full-resolution tensor. This is why DDPM is slow and memory-hungry.

**Key observation (Rombach et al. 2022):** Most of that pixel-space computation is redundant. Perceptually meaningful structure — objects, textures, composition — lives in a much lower-dimensional space. High-frequency pixel details can be handled separately.

---

### The VAE — Compressing to Latent Space

A **Variational Autoencoder (VAE)** is trained to compress images into a compact latent representation and reconstruct them back:

```
Image x ∈ ℝ^{H×W×3}
      ↓  Encoder E
Latent z ∈ ℝ^{h×w×c}     ← diffusion happens here
      ↓  Decoder D
Image x̂ ∈ ℝ^{H×W×3}
```

For Stable Diffusion v1, the compression factor is **f = 8** spatially:

$$512 \times 512 \times 3 \xrightarrow{E} 64 \times 64 \times 4$$

Dimensionality reduction:

$$\frac{512 \times 512 \times 3}{64 \times 64 \times 4} = \frac{786{,}432}{16{,}384} = 48\times \text{ smaller}$$

In practice with the full compute cost, this translates to **~64× faster** diffusion training and inference.

---

### VAE Training Objective

The VAE is trained independently with a combination of losses:

$$\mathcal{L}_{VAE} = \underbrace{\|x - D(E(x))\|^2}_{\text{reconstruction}} + \underbrace{\lambda_{KL} \cdot D_{KL}(q(z|x) \| \mathcal{N}(0,I))}_{\text{regularity}} + \underbrace{\lambda_{perc} \cdot \mathcal{L}_{LPIPS}}_{\text{perceptual}} + \underbrace{\lambda_{adv} \cdot \mathcal{L}_{GAN}}_{\text{adversarial (patch discriminator)}}$$

- **Reconstruction loss** — pixel-accurate recovery
- **KL divergence** — keeps the latent space compact and continuous (makes sampling possible)
- **Perceptual loss (LPIPS)** — matches high-level features, not just pixels
- **Adversarial loss** — a patch-GAN discriminator forces sharp, realistic textures

The KL weight $\lambda_{KL}$ is kept very small (e.g. $10^{-6}$) in LDM — this is a near-deterministic autoencoder, almost a VQ-VAE. The latent space is nearly regular but not strongly constrained.

---

### What the Latent Space Looks Like

The encoder $E$ maps an image to a distribution:

$$q(z \mid x) = \mathcal{N}(\mu_E(x),\ \sigma_E(x)^2 \mathbf{I})$$

A sample: $z = \mu_E(x) + \sigma_E(x) \odot \epsilon,\ \epsilon \sim \mathcal{N}(0, \mathbf{I})$

The 4-channel latent $z \in \mathbb{R}^{64 \times 64 \times 4}$ encodes:
- **Spatial structure** — preserved by the spatial dimensions 64×64
- **Semantic content** — packed into the 4 channels
- **Not directly interpretable** — but the diffusion model learns to navigate this space

---

### The Full LDM Architecture

```
                    ┌─────────────────────────────────┐
                    │         LATENT SPACE             │
                    │                                  │
  Image x           │  z_T ~ N(0,I)                   │
     ↓ Encoder E    │       ↓                          │
     z_0            │  Denoising U-Net (small!)        │
                    │  + Cross-Attention(c)  ← CLIP    │
                    │       ↓                          │
                    │  z_0 (predicted)                 │
                    └─────────────────────────────────┘
                              ↓ Decoder D
                          Image x̂
```

The U-Net now operates on $64 \times 64 \times 4$ instead of $512 \times 512 \times 3$. This is why the SD U-Net is **orders of magnitude smaller** than a pixel-space model of comparable output quality.

---

### Two-Stage Training (Critical Interview Point)

LDM training is **decoupled** into two completely separate stages:

**Stage 1 — Train the VAE** (once, reused forever):
- Learn $E$ and $D$ on large image datasets
- Goal: near-perfect reconstruction with a compact, regular latent space

**Stage 2 — Train the Diffusion Model in Latent Space**:
- Freeze $E$ and $D$ entirely
- Run the full DDPM/DDIM forward/reverse process on $z = E(x)$
- Train U-Net to predict $\epsilon$ in latent space

$$\mathcal{L}_{LDM} = \mathbb{E}_{z \sim E(x),\, \epsilon,\, t} \left[ \lVert \epsilon - \epsilon_\theta(z_t, t, \mathbf{c}) \rVert^2 \right]$$

The VAE **never sees noise** — it only processes clean images. The diffusion model **never sees pixels** — it only processes latents.

---

### Why This Works: The Perceptual Compression Hypothesis

Rombach et al. empirically showed that generative model learning has two phases:

```
Training compute →

Phase 1 (early): Model learns perceptual compression
                 (what is a face, what is a tree)

Phase 2 (late):  Model learns fine-grained detail
                 (exact pixel values, textures)
```

By offloading Phase 1 to the VAE, the diffusion model can **skip straight to learning semantics**. This is why LDM trains faster and generalizes better per FLOP.

---

### Stable Diffusion Versions: Latent Space Evolution

| Model | Latent dim | Channels | VAE type | Notes |
|---|---|---|---|---|
| SD v1.x | 64×64 | 4 | KL-VAE | Original LDM |
| SD v2.x | 64×64 | 4 | KL-VAE (improved) | Better text encoder (OpenCLIP) |
| SDXL | 128×128 | 4 | KL-VAE | Higher res latents, 2-stage U-Net |
| SD3 / Flux | 128×128 | 16 | Improved VAE | DiT backbone, more channels |

SDXL doubled the latent resolution (128×128) for sharper outputs while still being far cheaper than 1024×1024 pixel-space diffusion.

---

### The VAE Bottleneck Problem

A known limitation: the VAE is the **quality ceiling**. Any detail the encoder discards is unrecoverable by the diffusion model. Common artifacts:

- **Text rendering failure** — SD v1/v2 cannot generate legible text because CLIP+VAE can't preserve character-level detail at 8× compression
- **Fine detail loss** — hair, fingers, small objects get blurred at high compression
- **Color shifting** — the 4-channel latent sometimes produces hue drift

SD3 / Flux address this by using **16-channel latents** and an improved VAE with lower perceptual loss — more capacity per spatial location.

---

> **Key interview insight:** Latent Diffusion is essentially a **separation of concerns**: the VAE handles perceptual compression (a solved problem), and the diffusion model handles generative modeling (the hard problem). This decomposition is what made high-resolution diffusion models practical. The 64× speedup is not magic — it's the direct consequence of reducing the dimensionality of the space the diffusion process operates in from ~786K to ~16K.

---

## Level 6: Score Matching — The Theoretical SDE/ODE Framework

### What is a Score Function?

The **score function** of a probability distribution $p(x)$ is the gradient of its log-density with respect to $x$:

$$s(x) = \nabla_x \log p(x)$$

Intuition: it's a **vector field** that points toward regions of higher probability — toward the data manifold.

```
Low density region          High density region
     ·  →  →  →  →  →  ●●●●●
     ·  →  →  →  →  →  ●●●●●
     ·  →  →  →  →  →  ●●●●●
     ← score vectors point toward the data
```

If you know the score everywhere, you can **navigate toward samples** from $p(x)$ using **Langevin dynamics**:

$$x_{i+1} = x_i + \frac{\eta}{2} \nabla_x \log p(x_i) + \sqrt{\eta}\, \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \mathbf{I})$$

This is stochastic gradient ascent on the log-density — it converges to samples from $p(x)$.

---

### The Problem: We Don't Know $p(x)$

Real data distributions (natural images) are unknown. We only have samples. **Score matching** (Hyvärinen 2005) trains a neural network $s_\theta(x) \approx \nabla_x \log p(x)$ without ever knowing $p(x)$ explicitly.

**Denoising Score Matching (Vincent 2011):** Rather than matching the score of $p(x)$ directly (which requires computing a partition function), match the score of a **noised** distribution:

$$\mathcal{L}_{DSM} = \mathbb{E}_{x \sim p(x),\, \tilde{x} \sim q(\tilde{x}|x)} \left[ \lVert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x} \mid x) \rVert^2 \right]$$

For Gaussian noise $q(\tilde{x} \mid x) = \mathcal{N}(\tilde{x};\ x, \sigma^2 \mathbf{I})$:

$$\nabla_{\tilde{x}} \log q(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^2} = \frac{\epsilon}{\sigma}$$

So the target score is just the **normalized noise direction**. And this is *exactly* what DDPM's $\epsilon_\theta$ predicts — the connection is direct:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

**DDPM is denoising score matching.** The noise prediction network $\epsilon_\theta$ is a score network in disguise.

---

### The SDE Framework (Song et al. 2021)

Yang Song's landmark paper unified all diffusion models under a single continuous-time SDE framework.

**Forward SDE** — gradually corrupts data to noise:

$$dx = f(x, t)\, dt + g(t)\, dW$$

Where:
- $f(x, t)$ — drift coefficient (deterministic part)
- $g(t)$ — diffusion coefficient (noise scale)
- $dW$ — Wiener process (Brownian motion increment)

**Reverse SDE** — goes from noise back to data (Anderson 1982):

$$dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{W}$$

The reverse SDE requires the **score** $\nabla_x \log p_t(x)$ at every noise level $t$. Train a neural network $s_\theta(x, t)$ to estimate this — then you can run the reverse SDE to generate samples.

---

### DDPM and SMLD as Special Cases

| Method | SDE type | $f(x,t)$ | $g(t)$ |
|---|---|---|---|
| **DDPM** | VP-SDE (Variance Preserving) | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)}$ |
| **SMLD / NCSN** | VE-SDE (Variance Exploding) | $0$ | $\sqrt{\frac{d[\sigma^2(t)]}{dt}}$ |
| **Sub-VP SDE** | Between VP and VE | modified | modified |

**VP-SDE** (DDPM continuous limit): The noise schedule $\beta_t$ becomes a continuous function $\beta(t)$. As $T \to \infty$, DDPM's discrete Markov chain converges to this SDE. All the math stays the same.

**VE-SDE** (NCSN/Score Matching): The signal variance stays fixed, noise variance explodes. Easier to train but numerically less stable.

---

### The Probability Flow ODE — The Key to Fast Sampling

For every SDE, there exists a corresponding **deterministic ODE** with the same marginal distributions $p_t(x)$:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)$$

This is the **Probability Flow ODE**. It has no stochastic term but traces identical trajectories in distribution.

**Why this matters:**
1. ODEs can be solved with high-order numerical methods (fewer steps)
2. The mapping $x_T \leftrightarrow x_0$ is **deterministic** and **invertible**
3. This is exactly what DDIM is — a first-order solver of the probability flow ODE

$$\underbrace{\text{DDIM}}_{\text{1st order}} \subset \underbrace{\text{PNDM, DPM-Solver}}_{\text{higher order}} \subset \underbrace{\text{Probability Flow ODE solvers}}_{\text{general framework}}$$

---

### Three Equivalent Parameterizations of the Score Network

Given the score-noise relationship, you can train the network to predict three equivalent things:

| Prediction target | Symbol | Conversion to score |
|---|---|---|
| **Noise** (DDPM default) | $\epsilon_\theta$ | $s_\theta = -\epsilon_\theta / \sqrt{1-\bar\alpha_t}$ |
| **Score directly** | $s_\theta$ | $s_\theta$ |
| **Clean image** ($x_0$) | $x_\theta$ | $s_\theta = -(x_t - \sqrt{\bar\alpha_t}\, x_\theta) / (1-\bar\alpha_t)$ |
| **Velocity** (flow matching) | $v_\theta$ | $s_\theta = -(v_\theta + \sqrt{\bar\alpha_t}\,\epsilon) / \sqrt{1-\bar\alpha_t}$ |

These are all mathematically equivalent. The **velocity parameterization** is used in modern models (Stable Diffusion 3, Flux) because it has better conditioning near $t=0$ and $t=T$.

---

### Flow Matching — The Latest Generalization (Lipman et al. 2022)

Flow Matching replaces the SDE/score framework with **Continuous Normalizing Flows (CNFs)**:

Instead of learning a score, learn a **velocity field** $v_\theta(x, t)$ that defines the ODE:

$$\frac{dx}{dt} = v_\theta(x, t)$$

With a simple **linear interpolation** path between data $x_0$ and noise $x_1 \sim \mathcal{N}(0,I)$:

$$x_t = (1-t)\, x_0 + t\, x_1, \qquad t \in [0, 1]$$

The target velocity is just:

$$v^*(x_t, t) = x_1 - x_0$$

**Training loss:**

$$\mathcal{L}_{FM} = \mathbb{E}_{t,\, x_0,\, x_1} \left[ \lVert v_\theta(x_t, t) - (x_1 - x_0) \rVert^2 \right]$$

This is **simpler than DDPM** — straight-line paths from noise to data, no complex noise schedules.

**Why Flow Matching wins:**

```
DDPM path:  curved, complex, requires 1000 steps
Flow Match: straight line, fewer steps, better quality

x_1 (noise)
    ●
     \  ← DDPM curved path
      \  ___
       \/   \
       /\    ● x_0 (data)
      /  \__/
     ● ← Flow Matching straight path
```

SD3 and Flux use **Rectified Flow** (a variant) — this is why they achieve better quality at fewer steps than older SD versions.

---

### The Full Theoretical Lineage

```
Score Matching (Hyvärinen 2005)
    ↓
Denoising Score Matching (Vincent 2011)
    ↓
NCSN — Noise Conditional Score Networks (Song & Ermon 2019)
    ↓
DDPM (Ho et al. 2020) ←——→ NCSN++ (Song et al. 2020)
    ↓
Score SDE Framework — unifies everything (Song et al. 2021)
    ↓                         ↓
  VP-SDE (DDPM)            VE-SDE (NCSN)
    ↓
Probability Flow ODE → DDIM, DPM-Solver, PNDM
    ↓
Flow Matching / Rectified Flow (2022–2023) → SD3, Flux
```

---

> **Key interview insight:** The SDE framework reveals that **the score function is the only thing that matters**. Every diffusion model — regardless of architecture, noise schedule, or sampler — is fundamentally estimating $\nabla_x \log p_t(x)$. The forward process defines what distributions to estimate the score of, and the reverse process/ODE solver defines how to use those estimates to generate samples. Flow Matching simplifies the path geometry, making the score easier to estimate and the ODE faster to solve.

---

## Level 7: Vision Applications — Inpainting, Super-Resolution, Medical Imaging, Depth Estimation, Video Generation

### Overview

Each application exploits a different property of the generative process.

---

### 1. Inpainting — Filling Missing Regions

**Problem:** Given an image $x$ with a binary mask $m$ (1 = known, 0 = missing), generate a plausible completion that is coherent with the known region.

**Naive approach — RePaint (Ho et al. 2022):**

At each reverse step, force the known region to stay consistent with the original image:

$$x_{t-1}^{\text{known}} = \sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1-\bar{\alpha}_{t-1}}\, \epsilon$$
$$x_{t-1}^{\text{unknown}} = \text{reverse\_step}(x_t, t)$$
$$x_{t-1} = m \odot x_{t-1}^{\text{known}} + (1 - m) \odot x_{t-1}^{\text{unknown}}$$

**Problem with naive approach:** The known and unknown regions are denoised independently — they become inconsistent at boundaries (seam artifacts).

**RePaint fix:** After each reverse step, re-noise and re-denoise multiple times (`resample_steps`) at the boundary — forces the network to harmonize the regions before moving to the next timestep.

```
For each timestep t:
  Repeat r times:
    1. Merge known (noised) + unknown (denoised)
    2. Re-noise slightly → x_{t+1}
    3. Denoise again → x_t
  Move to t-1
```

**Production approach — Stable Diffusion Inpainting:**
Fine-tune the U-Net with an extra 5-channel input: 4 latent channels + 1 mask channel. The model is explicitly trained on masked data and learns to attend to the mask — no post-hoc resampling needed, much faster.

$$\epsilon_\theta(z_t, t, \mathbf{c}, z_{\text{masked}}, m)$$

**Applications:** Photo editing (remove objects), medical image restoration, video frame reconstruction, satellite image gap filling.

---

### 2. Super-Resolution — Hallucinating Detail

**Problem:** Given a low-resolution image $x_{LR}$, generate a high-resolution version $x_{HR}$ with plausible high-frequency detail.

This is **ill-posed** — many HR images can correspond to one LR image. Diffusion models embrace this by sampling from $p(x_{HR} \mid x_{LR})$ — producing diverse, plausible reconstructions.

**SR3 (Saharia et al. 2021):** Condition the reverse diffusion on $x_{LR}$:

$$\epsilon_\theta(x_t, t, x_{LR}^{\uparrow})$$

Where $x_{LR}^{\uparrow}$ is the bicubic-upsampled LR image, concatenated channel-wise to $x_t$ as input. The model learns to add realistic high-frequency detail consistent with the LR structure.

**Cascaded Diffusion Models (Ho et al. 2022):**

```
z ~ N(0,I)
  → Diffusion Model 1 → 64×64 image
  → Diffusion Model 2 (conditioned on 64×64) → 256×256
  → Diffusion Model 3 (conditioned on 256×256) → 1024×1024
```

Each stage is a separate diffusion model trained to upsample. This is the architecture behind **DALL·E 2** and **Imagen**.

**Why diffusion beats GAN-SR:**
- GANs (ESRGAN, Real-ESRGAN) produce sharp but often hallucinated textures that look unnatural under scrutiny
- Diffusion SR maintains semantic consistency because the reverse process is guided by the LR condition at every step
- Perceptual metrics (FID, LPIPS) favor diffusion; PSNR/SSIM may favor regression models (they optimize MSE, which blurs)

**Applications:** Medical imaging (MRI, CT upscaling), satellite imagery, surveillance, film restoration.

---

### 3. Medical Imaging — Where Diffusion Excels

Medical imaging has unique requirements: **trustworthy uncertainty**, **data scarcity**, **privacy constraints**. Diffusion models address all three.

#### 3a. Anomaly Detection

Train a diffusion model on **only healthy images**. At inference:

1. Corrupt a test image to timestep $t^*$ (partial noising)
2. Reconstruct via reverse diffusion
3. Compute pixel-wise difference: $\delta = |x_0 - x_{\text{reconstructed}}|$

Healthy regions reconstruct accurately ($\delta \approx 0$). Anomalous regions (tumors, lesions) are "corrected" toward the healthy manifold ($\delta > 0$) — the difference map is the anomaly score.

$$\text{Anomaly map} = \| x_0 - \hat{x}_0^{(t^*)} \|$$

The choice of $t^*$ controls sensitivity: small $t^*$ catches fine anomalies, large $t^*$ catches structural ones.

#### 3b. Synthetic Data Generation (Privacy-Preserving)

Train on real patient scans → generate synthetic scans → share synthetic dataset publicly without privacy risk. Used for:
- Training downstream models without data sharing agreements
- Augmenting rare disease datasets (few-shot learning)
- Class balancing (generate more of underrepresented pathologies)

#### 3c. Image-to-Image Translation

MRI → CT synthesis, T1 → T2 translation, PET → MRI synthesis. Avoids costly multi-modal acquisition. Conditioned diffusion (via cross-attention or concatenation) learns the mapping between modalities.

#### 3d. Reconstruction from Undersampled Measurements

In MRI, acquiring fewer k-space measurements = faster scans but aliased images. Diffusion models reconstruct from highly undersampled k-space by imposing data consistency at each reverse step:

$$\hat{x}_{t-1} = \text{reverse\_step}(x_t) \quad \text{then project onto} \quad \{x : \mathcal{A}(x) = y\}$$

Where $\mathcal{A}$ is the MRI forward operator and $y$ are the measurements. This enables **4–8× faster MRI scans** with diagnostic-quality reconstructions.

---

### 4. Depth Estimation — Marigold

**Problem:** Monocular depth estimation is ill-posed. Classical methods (MiDaS, DPT) produce deterministic outputs. Diffusion enables **probabilistic** depth estimation.

**Marigold (Ke et al. 2024):** Fine-tune Stable Diffusion for depth estimation by treating the depth map as the "image" to generate, conditioned on the RGB input:

```
RGB image x  →  VAE encoder  →  condition c
z_T ~ N(0,I)  →  Denoising U-Net(z_t, t, c)  →  z_0  →  VAE Decoder  →  Depth map
```

**Key insight:** Stable Diffusion's U-Net has learned extremely rich visual priors from internet-scale data. Fine-tuning it for depth requires **very little data** (synthetic renders) because the visual understanding is already there — you're just redirecting the output.

**Advantages over discriminative methods:**
- Uncertainty estimation: run multiple samples, variance = confidence
- Works zero-shot on novel domains (medical, satellite, microscopy)
- Affine-invariant depth (relative, not metric) — robust to scale ambiguity

**GeoWizard, StableNormal:** Same paradigm applied to surface normals and geometry estimation.

---

### 5. Video Generation — Temporal Diffusion

Extending image diffusion to video requires handling **temporal consistency** — frames must be coherent across time.

#### 5a. Architecture Approaches

**Inflate 2D → 3D (Make-A-Video, Imagen Video):**

Take a pretrained image U-Net and add **temporal attention** layers:

```
Spatial Self-Attention (per frame, pretrained, frozen)
       +
Temporal Self-Attention (across frames, newly added, trained)
```

Temporal attention treats the frame sequence as a 1D sequence of tokens, attending across time. Spatial layers handle per-frame quality; temporal layers handle consistency.

#### 5b. DiT for Video — Sora's Architecture

**Sora (OpenAI 2024)** uses a **Video DiT**:

1. Encode video frames with a VAE → latent sequence $z \in \mathbb{R}^{T \times h \times w \times c}$
2. Patchify spatially AND temporally → sequence of 3D tokens
3. Apply full transformer with 3D attention (space + time jointly)
4. Condition on text via cross-attention

The key innovation: **variable-length, variable-resolution** training. The same model handles 1s clips and 60s clips, 480p and 1080p, by treating them all as sequences of different lengths.

#### 5c. Temporal Consistency Strategies

| Strategy | Method | Used in |
|---|---|---|
| Temporal attention | Attend across frame tokens | Make-A-Video, Sora |
| Optical flow warping | Warp previous frame as condition | FILM, some SR models |
| Noise sharing | Correlate noise across frames | AnimateDiff |
| Autoregressive | Generate frames sequentially | Some early video models |

**AnimateDiff:** Inserts a **motion module** (temporal attention) into SD U-Net. Fine-tune only the motion module on video data. Existing SD LoRAs/checkpoints work unchanged — plug-and-play video for any SD model.

#### 5d. Video Editing — TokenFlow

**TokenFlow (2023):** Edit a video by propagating edits through DDIM inversion. Key insight: real video has consistent self-attention patterns across frames. Enforce that edited frames share the same attention keys/values → temporal consistency without per-frame degradation.

---

### 6. The Unified Framework — Any Vision Task as Conditional Generation

| Task | Condition $c$ | Output |
|---|---|---|
| Image segmentation | RGB image | Segmentation mask |
| Optical flow | Frame pair | Flow field |
| Pose estimation | RGB image | Keypoint heatmaps |
| Image colorization | Grayscale image | Color image |
| Shadow removal | Image with shadow | Shadow-free image |
| Deblurring | Blurry image | Sharp image |
| HDR reconstruction | LDR image | HDR image |

**Palette (2022):** A single diffusion model trained on all image-to-image translation tasks simultaneously — demonstrates that diffusion models are **general-purpose vision operators**.

---

### Practical Considerations Summary

| Application | Key challenge | Diffusion solution |
|---|---|---|
| Inpainting | Boundary coherence | RePaint resampling / masked fine-tuning |
| Super-resolution | Hallucination fidelity | LR concatenation conditioning |
| Medical | Uncertainty quantification | Ensemble of samples |
| Depth | Scale ambiguity | Affine-invariant output + fine-tuned priors |
| Video | Temporal consistency | Temporal attention / 3D DiT |

---

> **Key interview insight:** Diffusion models dominate vision tasks not because they were designed for each task, but because they are **universal conditional density estimators**. Any task of the form "given observation $y$, generate plausible $x$ consistent with $y$" can be framed as conditional diffusion. The score function $\nabla_x \log p(x \mid y)$ decomposes as $\nabla_x \log p(x) + \nabla_x \log p(y \mid x)$ — an unconditional prior plus a likelihood term. This Bayesian decomposition is why pre-trained diffusion models generalize so well to downstream vision tasks with minimal fine-tuning.

