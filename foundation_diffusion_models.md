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
