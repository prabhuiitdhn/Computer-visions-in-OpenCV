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

