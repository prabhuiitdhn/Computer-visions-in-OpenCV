# Image Restoration & Upscaling — Deep Conceptual Notes

Companion notes for Senior CV Researcher interview prep.
Topics: traditional restoration (denoising, deblurring, inpainting), upscaling (interpolation, super-resolution), frequency domain methods, and deep learning approaches.

---

## Image Restoration & Upscaling: Foundational Understanding

---

### What Is Image Restoration?

A captured image is rarely a perfect record of the scene. Between the real world and the final pixel values, **degradations** occur:

```
Real world scene
      ↓
  Optical system (lens blur, aberrations)
      ↓
  Motion (camera shake, moving subject)
      ↓
  Sensor (noise, limited dynamic range)
      ↓
  Transmission/Storage (compression artifacts, packet loss)
      ↓
Degraded observed image g(x,y)

Goal of restoration: recover f(x,y) — the clean original
```

Image restoration is the task of **inverting these degradations** — going from the observed corrupted image back toward the true clean image.

---

### The Universal Image Degradation Model

Every restoration problem can be expressed as one equation:

$$g = \mathcal{H}(f) + \eta$$

Where:
- $f$ = original clean image (what we want)
- $g$ = observed degraded image (what we have)
- $\mathcal{H}$ = degradation operator (blur, downsampling, masking, etc.)
- $\eta$ = noise (random corruption added on top)

This is the **forward model** — it describes how damage happens. Restoration is solving the **inverse problem**: given $g$, find $f$.

```
Forward (easy — damage is simple):    f → g = H(f) + η
Inverse (hard — many f's fit same g): g → f = ?
```

---

### Why Restoration Is an Ill-Posed Inverse Problem

The inverse problem has no unique solution in general — this is called being **ill-posed** (Hadamard's definition):

```
Well-posed problem requires ALL THREE:
  1. Solution EXISTS
  2. Solution is UNIQUE
  3. Solution depends CONTINUOUSLY on data (small change in g → small change in f)

Restoration violates condition 2 (uniqueness):
  Many different clean images f₁, f₂, f₃...
  when blurred and noised, all produce the same observed g.

Example: blurred photo of a cat
  Could be: sharp cat image → blurred
  Could also be: slightly different sharp cat → blurred differently
  Both fit the same degraded image g.
```

This is why restoration requires **regularization** — additional assumptions about what the true image should look like, to pick ONE solution from the infinite set of possibilities.

---

### The Three Types of Degradation You Must Know

#### Type 1: Noise

Random variation added to pixel values. No structural pattern — purely stochastic.

```
Observed:  g(x,y) = f(x,y) + η(x,y)
Degradation operator H = Identity (no blur, just noise)

Types of noise:
  Gaussian noise:  η ~ N(0, σ²)   — most common model
                   Camera sensor thermal noise
                   Additive, signal-independent

  Poisson noise:   η ~ Poisson(f)  — photon counting noise
                   Low-light photography
                   Signal-DEPENDENT: bright areas have more noise

  Salt-and-pepper: random pixels set to 0 or 255
                   Transmission errors, dead sensor pixels

  Speckle:         multiplicative noise, g = f · η
                   Radar, ultrasound imagery
```

Gaussian noise visualization:
```
Original:     50  50  50  50
              50  50  50  50    (uniform gray patch)
              50  50  50  50

After noise:  47  53  51  48
              52  49  54  50    (same patch, now randomly perturbed)
              48  51  47  53
```

---

#### Type 2: Blur (Convolution Degradation)

Spatial averaging — each pixel becomes a weighted sum of its neighbors. Information is **spread out** but not (immediately) lost.

$$g(x,y) = (h * f)(x,y) + \eta = \sum_{m,n} h(m,n) \cdot f(x-m, y-n) + \eta$$

Where $h$ is the **Point Spread Function (PSF)** or **blur kernel** — it describes how a single point of light spreads across the sensor.

```
Types of blur:

1. Gaussian blur (out-of-focus / lens blur):
   h = Gaussian kernel
   ┌───────────────┐
   │ 1  2  1       │
   │ 2  4  2  / 16 │   (3×3 example)
   │ 1  2  1       │
   └───────────────┘
   Circularly symmetric — same in all directions

2. Motion blur (camera shake / moving subject):
   h = line kernel along direction of motion
   h = [1 1 1 1 1] / 5   (5px horizontal motion)
   Directional — has an angle

3. Defocus blur:
   h = disk/pill-box kernel (flat circle)
   All pixels within radius r weighted equally
```

---

#### Type 3: Downsampling / Resolution Loss

The image is captured or stored at lower resolution than the scene contains:

```
Original: 8×8 image
      ↓  (2× downsampling: take every 2nd pixel)
Observed: 4×4 image

Degradation operator H = Downsample operator
No noise added, but information is IRREVERSIBLY lost
(aliasing — high frequencies fold into low frequencies)

This is the degradation that upscaling/super-resolution must reverse.
```

---

### What Is Upscaling?

Upscaling (or super-resolution) is a special case of restoration where the degradation is **downsampling**:

```
Forward model:  g = ↓(f) + η
                g = low-resolution image
                f = high-resolution image

Goal: recover f from g
      → produce more pixels than you have
      → recover high-frequency details that aren't in g
```

The fundamental question: **where do the extra pixels come from?**

```
Low-res image (4×4):           High-res target (8×8):

  A  B  C  D                  A  ?  B  ?  C  ?  D  ?
  E  F  G  H        →         ?  ?  ?  ?  ?  ?  ?  ?
  I  J  K  L                  E  ?  F  ?  G  ?  H  ?
  M  N  O  P                  ?  ?  ?  ?  ?  ?  ?  ?
                               ...

The ? pixels must be SYNTHESIZED. They don't exist in the input.
```

Traditional methods: **interpolate** (mathematically guess) the missing pixels.
Deep learning methods: **hallucinate** (predict) plausible high-frequency details from training data.

---

### The Signal Processing Foundation: Sampling Theory

To understand restoration and upscaling from first principles, you need **Nyquist-Shannon sampling theorem**:

> A signal can be perfectly reconstructed from its samples if the sampling rate is at least **twice** the highest frequency in the signal.

$$f_s \geq 2 \cdot f_{max} \quad \text{(Nyquist criterion)}$$

```
Spatial domain analogy for images:

Fine texture (high frequency):   alternating black/white stripes, 1px wide
Coarse texture (low frequency):  alternating black/white blocks, 10px wide

A camera sensor with pixel pitch p can capture frequencies up to:
  f_max = 1/(2p)  cycles per pixel  (Nyquist limit)

If the scene has finer details than this limit:
  → ALIASING: high frequencies fold back as false low-frequency patterns
  → jagged edges, moiré patterns, false colors in fine textures
```

This is why restoration/upscaling is hard: **aliased information cannot be recovered** — it's not just missing, it's corrupted into something else.

---

### The Frequency Domain View

Every image can be decomposed into sine waves of different frequencies (Fourier Transform):

```
Image = sum of:
  Low frequencies:  slow spatial variation → overall brightness, large regions
  Mid frequencies:  moderate variation → object edges, medium textures
  High frequencies: rapid spatial variation → fine details, sharp edges, noise

Fourier transform of a sharp image:
  |F(u,v)|²:  strong low-frequency components (center)
              weaker but present high-frequency components (edges of spectrum)

Fourier transform of a blurred image:
  |G(u,v)|²:  strong low-frequency components (preserved)
              HIGH FREQUENCIES ATTENUATED OR ZEROED OUT
              (this is what blur does in frequency domain: low-pass filter)
```

Each degradation has a clean frequency-domain interpretation:

```
Degradation:     Frequency domain effect:
──────────────   ────────────────────────
Gaussian blur    Multiply by Gaussian → attenuate high frequencies
Motion blur      Multiply by sinc function → periodic zeros
Noise            Add random values to ALL frequencies equally
Downsampling     Remove all frequencies above Nyquist limit
JPEG compression Quantize frequency coefficients → block artifacts
```

---

### The Restoration Taxonomy

```
IMAGE RESTORATION
│
├── DENOISING: remove η, H = Identity
│     Traditional: averaging, Gaussian filter, bilateral filter, NLM, BM3D
│     Deep learning: DnCNN, FFDNet, noise2noise
│
├── DEBLURRING: invert H (convolution), small or no noise
│     Traditional: Wiener filter, Richardson-Lucy, sparse priors
│     Deep learning: DeblurGAN, MPRNet
│
├── INPAINTING: fill missing regions (H = masking operator)
│     Traditional: diffusion-based, texture synthesis, exemplar-based
│     Deep learning: context encoder, LaMa, stable diffusion inpainting
│
└── SUPER-RESOLUTION: invert downsampling, recover H.F. details
      Traditional: bicubic, Lanczos, iterative back-projection
      Deep learning: SRCNN, ESRGAN, Real-ESRGAN, diffusion SR

IMAGE UPSCALING (subset of super-resolution)
│
├── Interpolation (no learned priors):
│     Nearest neighbor, bilinear, bicubic, Lanczos
│
└── Super-resolution (learned priors):
      Single-image SR (SISR): one LR image → HR image
      Multi-image SR: multiple LR images → one HR image
```

---

### The Core Trade-off in All Restoration: Fidelity vs Smoothness

Every restoration method must balance two competing objectives:

```
FIDELITY (data term):
  The restored image should match the observed degraded image
  after re-applying the degradation:
  min ‖H(f̂) - g‖²

  Benefit:  stays faithful to measurements
  Problem:  if you only minimize this, noise gets amplified

SMOOTHNESS (regularization term):
  The restored image should be "natural" — smooth, not noisy
  min ‖∇f̂‖² or ‖f̂‖_TV  (Total Variation)

  Benefit:  suppresses noise, produces clean images
  Problem:  over-smoothing destroys fine details and edges

COMBINED (Tikhonov regularization):
  min ‖H(f̂) - g‖² + λ‖∇f̂‖²

  λ controls the trade-off:
    λ → 0:  perfect fidelity, noisy
    λ → ∞:  over-smoothed, blurry
    λ just right: best of both → chosen by validation
```

This trade-off appears in **every** restoration method, traditional or deep learning.

---

### Why Traditional Methods Still Matter

```
Traditional methods:                Deep learning methods:
────────────────────                ──────────────────────
Mathematically interpretable        Black box
No training data needed             Requires large labeled datasets
Work on any image domain            May fail on out-of-distribution images
Computationally cheap               GPU-heavy at training
Controllable (tune λ)               Harder to control output
Provably optimal in some cases      Empirically better in many cases
Handle known noise models well      Learn to handle unknown degradations
```

---

### Key Metrics You Must Know

**PSNR (Peak Signal-to-Noise Ratio):**
$$\text{PSNR} = 10 \log_{10} \frac{255^2}{\text{MSE}}$$

where $\text{MSE} = \frac{1}{HW}\sum_{x,y}(f(x,y) - \hat{f}(x,y))^2$

```
PSNR interpretation (for 8-bit images):
  > 40 dB:  excellent, barely perceptible difference
  35–40 dB: good quality
  30–35 dB: acceptable, some artifacts visible
  < 30 dB:  noticeable degradation
```

**SSIM (Structural Similarity Index):**

$$\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Measures luminance, contrast, and structure separately. More aligned with human perception than PSNR.

```
SSIM range: [-1, 1]
  1.0:  identical images
  0.9+: visually similar
  < 0.7: clearly different
```

---

**Interview one-liner:**
> Image restoration inverts the degradation model $g = H(f) + \eta$ — recovering the clean image $f$ from the observed $g$. It is fundamentally ill-posed because many $f$'s produce the same $g$, requiring regularization (smoothness priors, sparsity, learned priors) to select the most plausible solution. Upscaling is the special case where $H$ is a downsampling operator — the extra pixels must be synthesized either by interpolation (traditional) or by predicting plausible high-frequency content from learned priors (deep learning).

---

## Why Restoration Is an Ill-Posed Inverse Problem: Hadamard's Definition

---

### Who Was Hadamard and What Did He Define?

Jacques Hadamard (1865–1963) was a French mathematician studying partial differential equations. In 1902 he formalized what it means for a problem to be **well-posed**:

> A mathematical problem is **well-posed** if and only if:
> 1. A solution **exists**
> 2. The solution is **unique**
> 3. The solution **depends continuously** on the data (small perturbations in input → small perturbations in output)

If ANY ONE of these three conditions fails, the problem is **ill-posed**.

Image restoration violates conditions **2 and 3** — and understanding exactly why is the foundation of everything in restoration theory.

---

### Condition 1: Does a Solution Exist?

For restoration: **usually yes** — for any observed degraded image $g$, there exists at least one clean image $f$ that could have produced it.

```
Observed blurry photo g
Does some clean f exist such that: blur(f) + noise = g ?
Answer: YES — at minimum, f = g itself (the blurry image)
        is a trivially valid "clean" image (just not a good one)
```

Condition 1 is the easiest to satisfy. Restoration doesn't fail here.

---

### Condition 2: Is the Solution Unique?

For restoration: **NO — catastrophically not unique**.

#### The Null Space Argument

Consider blurring as a linear operator $H$ (convolution with blur kernel $h$):

$$g = H f \quad \text{(ignoring noise for now)}$$

$H$ has a **null space** — a set of images $f_0$ such that $H f_0 = 0$ (they disappear completely after blurring):

```
What is in the null space of a blur operator?

Blur with a 3×3 Gaussian averages neighboring pixels.
High-frequency patterns that average to zero → vanish after blurring.

Example null space element (checkerboard pattern):
  +1  -1  +1  -1
  -1  +1  -1  +1
  +1  -1  +1  -1
  -1  +1  -1  +1

After Gaussian averaging: every pixel averages to ≈ 0
→ this pattern is INVISIBLE after blurring
```

Now if $f^*$ is the true clean image, then for ANY null space element $f_0$:

$$H(f^* + f_0) = Hf^* + Hf_0 = g + 0 = g$$

So $f^* + f_0$ is **also a valid solution** — it produces the same observed blurry image $g$.

```
The set of all valid solutions:
  {f^* + α·f₀ : α ∈ ℝ, f₀ ∈ null(H)}

This is an INFINITE-DIMENSIONAL AFFINE SUBSPACE.
Infinitely many clean images all produce the same blurry image.
```

Visually:
```
Clean image f*  (correct answer)
Clean image f* + small checkerboard (also valid)
Clean image f* + large checkerboard (also valid)
Clean image f* - medium checkerboard (also valid)
...

All of these blur to exactly the same g.
Which one should the restoration algorithm return?
```

There is no geometric basis to prefer one over another. **Uniqueness is broken.**

---

### Condition 3: Continuous Dependence on Data

This is the most dangerous failure — small changes in the observed image can cause **catastrophically large** changes in the reconstruction.

#### The Amplification Problem

Blurring multiplies each frequency by the blur's frequency response $H(u,v)$:

$$G(u,v) = H(u,v) \cdot F(u,v)$$

The naive inverse (divide by $H$):

$$\hat{F}(u,v) = \frac{G(u,v)}{H(u,v)}$$

The problem: $H(u,v)$ is **very small** (near zero) at high frequencies:

```
Gaussian blur frequency response H(u,v):

Low frequencies:   H ≈ 1.0     (pass through unchanged)
Mid frequencies:   H ≈ 0.3     (attenuated)
High frequencies:  H ≈ 0.001   (almost zero)
Very high freqs:   H ≈ 0.00001 (essentially zero)
```

Now add a tiny amount of noise $\eta$ to the observed image:

$$\hat{F}_{noisy}(u,v) = F(u,v) + \frac{N(u,v)}{H(u,v)}$$

The noise gets **divided by the tiny $H$ value** at high frequencies:

```
Noise level:         σ_N = 0.01  (1% noise in observed image)

After inversion:
  Low freq noise:    0.01 / 1.0     = 0.01    (unchanged)
  Mid freq noise:    0.01 / 0.3     = 0.033   (3× amplified)
  High freq noise:   0.01 / 0.001   = 10.0    (1000× amplified!)
  Very high noise:   0.01 / 0.00001 = 1000.0  (100,000× amplified!!)
```

A 1% noise perturbation in the input becomes **100,000%** in the reconstruction at high frequencies. This is **discontinuous dependence on data** — Hadamard's third condition violated.

```
Concrete example:

Input g₁:  exact blurry image (no noise)
Input g₂:  blurry image + tiny grain (σ=0.01, imperceptible)

Reconstruction from g₁:  perfect sharp image ✓
Reconstruction from g₂:  completely noise-dominated garbage ✗

‖g₁ - g₂‖ = 0.01 (tiny)   but   ‖f̂₁ - f̂₂‖ = 1000 (enormous)
```

---

### The Geometric Picture: Ill-Posedness as a "Flat" Forward Map

```
Space of all clean images F         Space of all blurred images G
(high-dimensional)                  (lower-dimensional, compressed)

                H
F  ──────────────────────────→  G

The map H "collapses" many points in F to the same point in G:
  f₁, f₂, f₃, ... → same g

This is like projecting 3D space to 2D:
  Many 3D points → same 2D shadow

Inverting this means: given a 2D shadow, find the 3D object.
Impossible without additional info about the "height" (null space direction).
```

---

### What Regularization Does: Picking One Solution

**Without regularization** (naive inverse):
```
min ‖Hf - g‖²
Solution: infinitely many — any f in the fiber above g
Degenerate solution: amplifies noise catastrophically
```

**With Tikhonov regularization**:
```
min ‖Hf - g‖² + λ‖∇f‖²
      ↑               ↑
  data fidelity    smoothness prior

Picks the UNIQUE f that is consistent with g AND as smooth as possible.
The regularization term breaks the degeneracy → unique minimum.
```

Different priors → different "best" solutions:

```
Prior:                          Effect:
──────────────────────────────  ──────────────────────────────────
L2 smoothness ‖∇f‖²            Gaussian prior → smooth, blurry edges
Total Variation ‖∇f‖₁          Piecewise constant → sharp edges, flat regions
Sparsity in wavelet domain      Natural image prior → good texture/edge tradeoff
Learned deep network prior      Data-driven → best perceptual quality
```

---

### The Condition Number: Quantifying Ill-Posedness

$$\kappa(H) = \frac{\sigma_{max}}{\sigma_{min}}$$

where $\sigma_{max}$ and $\sigma_{min}$ are the largest and smallest singular values of $H$.

```
Well-posed problem:   κ ≈ 1      (small noise → small error)
Mildly ill-posed:     κ ≈ 100    (small noise → moderate error)
Severely ill-posed:   κ ≈ 10⁶   (small noise → enormous error)
Perfectly ill-posed:  κ = ∞      (H has zero singular values → null space exists)

Gaussian blur:         κ ≈ 10⁶ to 10¹²  (severely ill-posed)
Motion blur:           κ = ∞              (zeros in H(u,v) → infinite condition number)
```

---

### Three Restoration Problems: Degrees of Ill-Posedness

```
DENOISING (H = Identity):
  H has no null space (identity is invertible)
  Condition number = 1
  Degree of ill-posedness: MILD
  → easy to regularize, well-studied

DEBLURRING (H = convolution):
  H has near-zero values at high frequencies → near-null-space
  Condition number ≈ 10⁶ to ∞
  Degree of ill-posedness: SEVERE
  → extremely sensitive to noise

SUPER-RESOLUTION (H = downsample):
  H discards high frequencies entirely → true null space
  Condition number = ∞ (not invertible at all)
  Degree of ill-posedness: COMPLETE (truly ill-posed)
  → missing information must be INVENTED from priors
```

Super-resolution is the most ill-posed — the **information needed to reconstruct simply does not exist** in the measurement. It must be supplied entirely by priors.

---

### Why This Matters for Deep Learning Restoration

Deep networks don't make restoration well-posed — they implement a powerful learned **prior**:

```
Deep network restoration:
  The network learns: "Among all valid solutions for this g, pick
  the one most consistent with the distribution of natural images
  seen in training"

Failure mode:
  If g comes from a different distribution than training
  → the implicit prior is wrong → the network picks the
     wrong solution from the infinite valid set
```

This is why deep restoration models **fail on out-of-distribution inputs** — their regularization (prior) is calibrated to the training distribution.

---

### Summary: The Three Failures

```
Hadamard Condition      Status in Restoration    Why
────────────────────    ────────────────────────  ─────────────────────────────
1. Solution exists      ✓ Usually satisfied       Always some f consistent with g
2. Unique solution      ✗ FAILS                   Null space of H: f* + f₀ also valid
3. Continuous on data   ✗ FAILS BADLY             Small noise → enormous reconstruction error
                                                   (noise amplified by 1/H at high frequencies)

Consequence: regularization REQUIRED to:
  - Pick one solution from the infinite valid set (fix condition 2)
  - Suppress noise amplification (fix condition 3)
```

---

**Interview one-liner:**
> Restoration is ill-posed in Hadamard's sense because the forward operator $H$ has a null space (violating uniqueness — $f^* + f_0$ produces the same $g$ for any $f_0 \in \text{null}(H)$) and near-zero singular values (violating continuous dependence — tiny noise $\eta$ gets amplified by $1/H(u,v)$ at high frequencies, causing catastrophic instability in the naive inverse). Regularization restores well-posedness by adding a prior that selects the unique "most natural" solution from the infinite valid set, controlling noise amplification through a fidelity-smoothness trade-off.

---

## Additive Noise vs Multiplicative Noise: The Fundamental Difference

---

### The Core Distinction in One Line

```
Additive noise:       g = f + η        noise is INDEPENDENT of signal
Multiplicative noise: g = f · η        noise SCALES with signal strength
```

The difference isn't just mathematical notation — it reflects completely different **physical mechanisms** of corruption, requires different **statistical models**, and demands different **restoration approaches**.

---

### Additive Noise: The "Background Hum"

#### Physical Origin

Additive noise comes from sources that exist **regardless of whether light is hitting the sensor**:

```
Sources of additive noise in cameras:

1. THERMAL (Johnson-Nyquist) noise:
   Electrons randomly move in sensor circuitry due to heat.
   Even in complete darkness, the sensor reads non-zero values.
   → This is why cameras have "dark current" — black frame ≠ zero

2. Read noise:
   Amplifier circuits introduce noise when reading out pixel values.
   Independent of how much light hit the pixel.

3. Quantization noise:
   Analog → digital conversion rounds to nearest integer level.
   The rounding error is bounded by ±0.5 LSB, independent of signal.

4. Transmission noise:
   Radio frequency interference, cable crosstalk.
   Noise signal superimposed on image signal.
```

#### The Mathematical Model

$$g(x,y) = f(x,y) + \eta(x,y)$$

Key property: **η is statistically independent of f**

```
Signal strength:   10    50    100   200
Noise (σ=5):      ±5    ±5    ±5    ±5

Signal-to-noise:   10/5  50/5  100/5  200/5
                   = 2   = 10  = 20   = 40

SNR IMPROVES with brighter signal → dark regions look noisier
```

This is why bright areas look clean but dark areas look grainy — **in additive noise, SNR is proportional to signal strength**.

#### Gaussian Additive Noise: The Standard Model

$$g(x,y) = f(x,y) + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2)$$

- Mean of noise: $E[\eta] = 0$ → on average, $g = f$ (noise is unbiased)
- Variance: $\text{Var}(\eta) = \sigma^2$ → same regardless of pixel brightness
- Distribution of observed pixel: $g \sim \mathcal{N}(f, \sigma^2)$

---

### Multiplicative Noise: The "Percentage Corruption"

#### Physical Origin

Multiplicative noise comes from sources that **interact with the signal itself** — corruption proportional to what was there:

```
Sources of multiplicative noise:

1. SPECKLE NOISE (coherent imaging):
   Radar, SAR, laser speckle, ultrasound.
   Coherent radiation scattered from many sub-resolution points
   INTERFERE constructively/destructively.
   Interference variation is proportional to surface reflectivity f.

2. FILM GRAIN (photographic film):
   Silver halide crystals have random sizes.
   Bright areas: more crystals develop → more grain variation
   Dark areas: fewer crystals → less grain

3. ATMOSPHERIC TURBULENCE (remote sensing):
   Fluctuations in atmospheric refractive index
   modulate the intensity of light: g(x,y) = f(x,y) · atm_fluctuation(x,y)
```

#### The Mathematical Model

$$g(x,y) = f(x,y) \cdot \eta(x,y)$$

Key property: **noise magnitude scales with the signal**

```
Signal strength:   10    50    100   200
Noise (30%):      ±3    ±15   ±30   ±60

Signal-to-noise:   10/3  50/15  100/30  200/60
                   = 3.3 = 3.3  = 3.3   = 3.3

SNR IS CONSTANT → bright and dark regions look equally noisy
(the PERCENTAGE of corruption is constant)
```

**Common multiplicative noise model — Speckle (Gamma distributed):**

$$g = f \cdot \eta, \quad \eta \sim \text{Gamma}(L, 1/L)$$

where $L$ = number of looks: $L=1$ (fully developed speckle), $L→\infty$ (noise averages out).

---

### Side-by-Side: The Critical Differences

| Property | Additive Noise | Multiplicative Noise |
|---|---|---|
| Model | $g = f + \eta$ | $g = f \cdot \eta$ |
| Independence | $\eta$ independent of $f$ | $\eta$ scales with $f$ |
| Dark regions | High relative noise (low SNR) | Same relative noise as bright |
| Bright regions | Low relative noise (high SNR) | Same relative noise as dark |
| SNR behavior | SNR ∝ $f$ (improves with brightness) | SNR ≈ constant |
| Noise variance in uniform region | Constant everywhere | Variance ∝ $f^2$ |
| Dominant domain | Natural photography, sensors | Radar, SAR, ultrasound, laser |
| Noise distribution | Gaussian (most common) | Gamma, Rayleigh, log-normal |

---

### Visual Intuition: What They Look Like

**Additive noise on a gradient (σ=10):**
```
True:    10   30   50   70   90   110   130   150
Noisy:    3   25   58   62   97   104   122   158  ← deviation ~10 everywhere

Dark region (10):    ±10 deviation → 100% perturbation (VERY NOISY)
Bright region (150): ±10 deviation →  6.7% perturbation (barely noticeable)
```

**Multiplicative noise on same gradient (30%):**
```
True:    10   30   50   70   90   110   130   150
Noisy:    7   37   42   77   83   126   119   172  ← deviation ∝ signal

Dark region (10):    ±3  deviation → 30% perturbation
Bright region (150): ±45 deviation → 30% perturbation  (same percentage!)
```

---

### The Logarithm Trick: Converting Multiplicative → Additive

Taking the **logarithm** of both sides:

$$g = f \cdot \eta \;\;\Rightarrow\;\; \log g = \log f + \log \eta$$

Let $G = \log g$, $F = \log f$, $N = \log \eta$:

$$G = F + N \quad \text{← now ADDITIVE!}$$

```
Restoration pipeline for multiplicative noise:
  1. Take log of observed image: G = log(g)
  2. Apply any additive denoising method to G → F̂
  3. Exponentiate back: f̂ = exp(F̂)
```

#### Why the Log Works

```
Multiplicative noise variance:
  Var(g) = f² · Var(η)   → scales with f² (signal-dependent)

After log transform (delta method):
  Var(log g) ≈ Var(η) / E[η]²   → CONSTANT (signal-independent)
```

The log transform **homogenizes** the noise variance — making it signal-independent, just like additive noise.

---

### Poisson Noise: The In-Between Case

$$g \sim \text{Poisson}(f), \quad E[g] = f, \quad \text{Var}(g) = f$$

Variance grows with signal (like multiplicative) but only as $f$, not $f^2$:

```
STD comparison at various signal levels:

Signal f:             10      100      400
Additive (σ=10):      10       10       10   (constant)
Poisson:            √10≈3.2  √100=10  √400=20  (grows as √f)
Multiplicative(30%):   3       30      120   (grows as f)
```

Poisson noise dominates in **photon-limited imaging** (astronomy, fluorescence microscopy, low-light cameras).

**Anscombe variance-stabilizing transform:**

$$z = 2\sqrt{g + 3/8}$$

After this: $z \approx \mathcal{N}(2\sqrt{f}, 1)$ — approximately Gaussian with **unit variance**. Then apply Gaussian denoising and invert.

---

### Restoration Strategies by Noise Type

#### For Additive Gaussian Noise:
```
Wiener filter (frequency domain):
  F̂(u,v) = G(u,v) · |F|² / (|F|² + σ²_noise)

Spatial domain: Gaussian filter, bilateral filter, NLM, BM3D
MAP optimization: min ‖g - f‖² + λ‖∇f‖²
```

#### For Multiplicative Speckle Noise:
```
Option 1: Log domain  → additive denoising → exp()
Option 2: Lee filter (classical SAR despeckling):
  f̂ = ḡ + k(g - ḡ),  k = σ²_f / (σ²_f + σ²_η·ḡ²)
  k→1 at edges (preserve), k→0 in flat regions (smooth)
Option 3: Deep learning SAR-CNN trained on simulated speckle
```

#### For Poisson Noise:
```
Option 1: Anscombe transform → Gaussian denoising → inverse Anscombe
Option 2: Richardson-Lucy algorithm (used in astronomy/microscopy):
  min Σ [f - g·log(f)] + λ R(f)  (Poisson negative log-likelihood)
```

---

### Why Noise Model Choice Matters

```
Wrong noise model → wrong restoration → worse than no restoration

Applying Gaussian denoising to SAR speckle (multiplicative):
  → edges smeared differently, flat regions wrong smoothness
  → noise partially remains due to model mismatch

Applying Lee filter to camera noise (additive):
  → dark areas over-smoothed (assumes dark = relatively noisier)
  → loss of shadow detail

Fluorescence microscopy with Gaussian filter (Poisson noise):
  → uniform smoothing ignores intensity-dependent noise
  → dim structures over-smoothed
  Anscombe + Gaussian + inverse: much better
```

---

### The Real World: Mixed Noise Models

Real cameras have **both** simultaneously:

$$g = \text{Poisson}(f) + \mathcal{N}(0, \sigma_{read}^2)$$

- **High light**: Poisson dominates (shot noise) → multiplicative character
- **Low light**: additive read noise dominates

Unified heteroscedastic Gaussian approximation (used in modern ISPs):

$$\text{Var}(g) \approx \alpha \cdot f + \beta$$

where $\alpha$ = Poisson gain, $\beta$ = read noise variance.

---

**Interview one-liner:**
> Additive noise ($g = f + \eta$) is signal-independent — the noise magnitude is constant regardless of pixel brightness, so dark regions appear noisier in relative terms (low SNR). Multiplicative noise ($g = f \cdot \eta$) scales with signal strength — noise magnitude is proportional to $f$, giving constant relative SNR across all intensities. The key practical tool for multiplicative noise is the log transform: $\log g = \log f + \log \eta$ converts it to an additive model, allowing all standard additive denoising methods (Wiener filter, BM3D, NLM) to be applied, followed by exponentiation back to the original domain.

---

*End of notes — continued in next session.*