"""
Image Thresholding Techniques using NumPy
Base image: BN5.png

Techniques implemented:
  1. Fixed (manual) thresholding
  2. Otsu's method (automatic)
  3. Adaptive mean thresholding
  4. Adaptive Gaussian thresholding
  5. Multi-level Otsu (3-class)
  6. Triangle method
  7. Hysteresis thresholding

All outputs saved as *_thresh.png in foundations/.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import os
import time

# ============================================================
# LOAD IMAGE
# ============================================================
img = np.array(Image.open("BN5.png"))
print(f"Loaded image: shape={img.shape}, dtype={img.dtype}")

OUTPUT_DIR = "."

# Downsample for speed
scale = 4
img_small = img[::scale, ::scale, :]
print(f"Working on downsampled image: {img_small.shape}")

# Convert to grayscale
gray = np.dot(img_small[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
print(f"Grayscale: shape={gray.shape}, range=[{gray.min()}, {gray.max()}]")


def save(name, arr):
    path = os.path.join(OUTPUT_DIR, name)
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")


save("grayscale_thresh.png", gray)


# ============================================================
# 1. FIXED (MANUAL) THRESHOLDING
# ============================================================
def fixed_threshold(image, T):
    """Binary threshold: 255 if pixel > T, else 0."""
    return np.where(image > T, 255, 0).astype(np.uint8)


print("\n========== IMAGE THRESHOLDING TECHNIQUES ==========")

t = time.time()
print("[1] Fixed Thresholding (T=127)...")
binary_fixed = fixed_threshold(gray, T=127)
save("fixed_thresh.png", binary_fixed)
print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# 2. OTSU'S METHOD (AUTOMATIC)
# ============================================================
def otsu_threshold(image):
    """
    Otsu's method: find threshold maximizing between-class variance.
    Returns (threshold, binary_image).
    """
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total = image.size
    prob = hist / total

    best_t = 0
    best_sigma_b = 0

    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    global_mean = cum_mean[-1]

    for t in range(256):
        w0 = cum_prob[t]
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[t] / w0
        mu1 = (global_mean - cum_mean[t]) / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
        if sigma_b > best_sigma_b:
            best_sigma_b = sigma_b
            best_t = t

    binary = np.where(image > best_t, 255, 0).astype(np.uint8)
    return best_t, binary


t = time.time()
print("[2] Otsu's Method...")
otsu_t, binary_otsu = otsu_threshold(gray)
save("otsu_thresh.png", binary_otsu)
print(f"    Otsu threshold = {otsu_t}")
print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# 3. ADAPTIVE MEAN THRESHOLDING
# ============================================================
def adaptive_mean_threshold(image, block_size, C):
    """
    Adaptive thresholding using local mean.
    T(x,y) = mean(block neighborhood) - C
    """
    pad = block_size // 2
    padded = np.pad(image.astype(np.float64), pad, mode='edge')
    h, w = image.shape
    shape = (h, w, block_size, block_size)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)
    local_mean = windows.reshape(h, w, -1).mean(axis=2)
    thresh_map = local_mean - C
    return np.where(image > thresh_map, 255, 0).astype(np.uint8)


t = time.time()
print("[3] Adaptive Mean Thresholding (block=25, C=10)...")
binary_adapt_mean = adaptive_mean_threshold(gray, block_size=25, C=10)
save("adaptive_mean_thresh.png", binary_adapt_mean)
print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# 4. ADAPTIVE GAUSSIAN THRESHOLDING
# ============================================================
def adaptive_gaussian_threshold(image, block_size, C, sigma=None):
    """
    Adaptive thresholding using Gaussian-weighted local mean.
    T(x,y) = gaussian_weighted_mean(block) - C
    """
    if sigma is None:
        sigma = block_size / 6.0
    pad = block_size // 2
    padded = np.pad(image.astype(np.float64), pad, mode='edge')
    h, w = image.shape

    # Build Gaussian kernel
    ax = np.arange(block_size) - pad
    xx, yy = np.meshgrid(ax, ax)
    gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    shape = (h, w, block_size, block_size)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)
    local_mean = np.einsum('ijkl,kl->ij', windows, gauss)
    thresh_map = local_mean - C
    return np.where(image > thresh_map, 255, 0).astype(np.uint8)


t = time.time()
print("[4] Adaptive Gaussian Thresholding (block=25, C=10)...")
binary_adapt_gauss = adaptive_gaussian_threshold(gray, block_size=25, C=10)
save("adaptive_gaussian_thresh.png", binary_adapt_gauss)
print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# 5. MULTI-LEVEL OTSU (3-CLASS)
# ============================================================
def multi_otsu_threshold(image, n_classes=3):
    """
    Multi-level Otsu: find (n_classes - 1) thresholds maximizing
    between-class variance. Supports 3 classes (2 thresholds).
    """
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total = image.size
    prob = hist / total
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    global_mean = cum_mean[-1]

    best_sigma = 0
    best_t1, best_t2 = 0, 0

    for t1 in range(1, 254):
        for t2 in range(t1 + 1, 255):
            w0 = cum_prob[t1]
            w1 = cum_prob[t2] - cum_prob[t1]
            w2 = 1.0 - cum_prob[t2]
            if w0 == 0 or w1 == 0 or w2 == 0:
                continue
            mu0 = cum_mean[t1] / w0
            mu1 = (cum_mean[t2] - cum_mean[t1]) / w1
            mu2 = (global_mean - cum_mean[t2]) / w2
            sigma = (w0 * (mu0 - global_mean)**2 +
                     w1 * (mu1 - global_mean)**2 +
                     w2 * (mu2 - global_mean)**2)
            if sigma > best_sigma:
                best_sigma = sigma
                best_t1, best_t2 = t1, t2

    out = np.zeros_like(image)
    out[image > best_t2] = 255
    out[(image > best_t1) & (image <= best_t2)] = 127
    return best_t1, best_t2, out


t = time.time()
print("[5] Multi-level Otsu (3-class)...")
t1_mo, t2_mo, binary_multi = multi_otsu_threshold(gray, n_classes=3)
save("multi_otsu_thresh.png", binary_multi)
print(f"    Thresholds: T1={t1_mo}, T2={t2_mo}")
print(f"    Time: {time.time()-t:.2f}s")


# ============================================================
# 6. TRIANGLE METHOD
# ============================================================
def triangle_threshold(image):
    """
    Triangle method: draw line from histogram peak to farthest end,
    find threshold where perpendicular distance is maximum.
    """
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))

    peak_idx = np.argmax(hist)
    peak_val = hist[peak_idx]

    # Determine which side is longer (search on the longer tail)
    left_dist = peak_idx - 0
    right_dist = 255 - peak_idx

    if left_dist > right_dist:
        x1, y1 = 0, float(hist[0])
        x2, y2 = peak_idx, float(peak_val)
        search_range = range(0, peak_idx)
    else:
        x1, y1 = peak_idx, float(peak_val)
        x2, y2 = 255, float(hist[255])
        search_range = range(peak_idx, 256)

    # Line: ax + by + c = 0
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = np.sqrt(a**2 + b**2)
    if norm == 0:
        return 127, np.where(image > 127, 255, 0).astype(np.uint8)

    best_t = 0
    best_dist = 0
    for i in search_range:
        dist = abs(a * i + b * float(hist[i]) + c) / norm
        if dist > best_dist:
            best_dist = dist
            best_t = i

    binary = np.where(image > best_t, 255, 0).astype(np.uint8)
    return best_t, binary


t = time.time()
print("[6] Triangle Method...")
tri_t, binary_tri = triangle_threshold(gray)
save("triangle_thresh.png", binary_tri)
print(f"    Triangle threshold = {tri_t}")
print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# 7. HYSTERESIS THRESHOLDING
# ============================================================
def hysteresis_threshold(image, low, high):
    """
    Hysteresis thresholding with 8-connectivity.
    Strong pixels (>= high): always kept.
    Weak pixels (>= low, < high): kept only if connected to strong.
    """
    strong = (image >= high)
    weak = (image >= low) & (image < high)
    out = np.zeros_like(image, dtype=np.uint8)
    out[strong] = 255

    h, w = image.shape
    changed = True
    while changed:
        changed = False
        padded = np.pad(out, 1, mode='constant', constant_values=0)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbor = padded[1+di:1+di+h, 1+dj:1+dj+w]
                promote = weak & (out == 0) & (neighbor == 255)
                if np.any(promote):
                    out[promote] = 255
                    weak[promote] = False
                    changed = True
    return out


t = time.time()
print("[7] Hysteresis Thresholding (low=80, high=160)...")
binary_hyst = hysteresis_threshold(gray, low=80, high=160)
save("hysteresis_thresh.png", binary_hyst)
print(f"    Time: {time.time()-t:.4f}s")


print("\n All thresholded images saved in foundations/ with *_thresh.png tag!")