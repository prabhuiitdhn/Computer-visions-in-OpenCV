"""
Linear and Non-Linear Image Filtering using NumPy
Base image: BN5.png

Linear Filters   -> Output = weighted sum of neighborhood (convolution)
Non-Linear Filters -> Output = non-linear function of neighborhood (median, min, max)

Vectorized with NumPy stride tricks for performance.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import os
import time

# ============================================================
# LOAD IMAGE using NumPy (via PIL for decoding PNG)
# ============================================================
img = np.array(Image.open("BN5.png"))  # shape: (H, W, 3), dtype: uint8, RGB
print(f"Loaded image: shape={img.shape}, dtype={img.dtype}")

OUTPUT_DIR = "."  # foundations folder


# ============================================================
# VECTORIZED 2D Convolution using stride tricks (Linear Filter)
# ============================================================
def convolve2d(image, kernel):
    """
    Vectorized 2D convolution using NumPy stride tricks.
    - image: 2D array (H, W)
    - kernel: 2D array (kH, kW), odd-sized
    Returns: convolved image, same size (edge-padded)
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(np.float64)
    h, w = image.shape
    # Create sliding window view — no data copied, overlapping views into same memory
    shape = (h, w, kh, kw)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)
    # Batched element-wise multiply + sum via einsum
    return np.einsum('ijkl,kl->ij', windows, kernel)


def apply_filter_rgb(image, kernel):
    """Apply convolution to each RGB channel independently."""
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(3):
        result[:, :, c] = convolve2d(image[:, :, c].astype(np.float64), kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# VECTORIZED Non-linear filter using stride tricks
# ============================================================
def nonlinear_filter(image, size, func):
    """
    Vectorized non-linear filter on single channel.
    - size: kernel size (odd)
    - func: np.median, np.min, or np.max
    """
    pad = size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge').astype(np.float64)
    h, w = image.shape
    shape = (h, w, size, size)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)
    flat_windows = windows.reshape(h, w, -1)
    return func(flat_windows, axis=2)


def apply_nonlinear_rgb(image, size, func):
    """Apply non-linear filter to each RGB channel."""
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(3):
        result[:, :, c] = nonlinear_filter(image[:, :, c].astype(np.float64), size, func)
    return np.clip(result, 0, 255).astype(np.uint8)


def save(name, arr):
    """Save NumPy array as PNG."""
    path = os.path.join(OUTPUT_DIR, name)
    Image.fromarray(arr).save(path)
    print(f"  Saved: {path}")

def bilateral_filter(image, diameter, sigma_color, sigma_space):
    """
    Bilateral filter for grayscale or RGB image (slow, reference implementation).
    - diameter: window size (odd)
    - sigma_color: filter sigma in color space
    - sigma_space: filter sigma in coordinate space
    """
    if image.ndim == 3:
        # Apply to each channel independently
        return np.stack([
            bilateral_filter(image[..., c], diameter, sigma_color, sigma_space)
            for c in range(image.shape[2])
        ], axis=-1)
    h, w = image.shape
    pad = diameter // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge').astype(np.float64)
    out = np.zeros_like(image, dtype=np.float64)
    # Precompute spatial Gaussian
    ax = np.arange(diameter) - pad
    xx, yy = np.meshgrid(ax, ax)
    spatial = np.exp(-(xx**2 + yy**2) / (2 * sigma_space**2))
    for i in range(h):
        for j in range(w):
            region = padded[i:i+diameter, j:j+diameter]
            center = padded[i+pad, j+pad]
            color = np.exp(-((region - center) ** 2) / (2 * sigma_color**2))
            weights = spatial * color
            weights_sum = np.sum(weights)
            out[i, j] = np.sum(region * weights) / weights_sum
    return np.clip(out, 0, 255).astype(np.uint8)

def morphological_filter(image, size, mode):
    """
    Morphological filter (erosion/dilation) for grayscale or RGB image.
    - size: structuring element size (odd)
    - mode: 'erosion' or 'dilation'
    """
    func = np.min if mode == 'erosion' else np.max
    if image.ndim == 3:
        out = np.stack([
            nonlinear_filter(image[..., c], size, func)
            for c in range(image.shape[2])
        ], axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)
    out = nonlinear_filter(image, size, func)
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# Downsample for faster processing (original is 2688x1792)
# ============================================================
scale = 4
img_small = img[::scale, ::scale, :]
print(f"Working on downsampled image: {img_small.shape}")
save("original_filter.png", img_small)


# ============================================================
# ===================== LINEAR FILTERS =======================
# ============================================================
print("\n========== LINEAR FILTERS (convolution-based) ==========")

# 1. MEAN (BOX) FILTER — averages neighborhood, blurs image
t = time.time()
print("[1] Mean (Box) Filter 5x5...")
k = 5
mean_kernel = np.ones((k, k), dtype=np.float64) / (k * k)
img_mean = apply_filter_rgb(img_small, mean_kernel)
save("mean_box_filter.png", img_mean)
print(f"    Time: {time.time()-t:.2f}s")

# 2. GAUSSIAN FILTER — weighted average, center has more weight
t = time.time()
print("[2] Gaussian Filter 5x5, sigma=1.0...")
def gaussian_kernel(size, sigma):
    """Generate 2D Gaussian kernel using NumPy."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

gauss_k = gaussian_kernel(5, sigma=1.0)
print(f"    Gaussian kernel (rounded):\n{np.round(gauss_k, 4)}")
img_gaussian = apply_filter_rgb(img_small, gauss_k)
save("gaussian_filter.png", img_gaussian)
print(f"    Time: {time.time()-t:.2f}s")

# 3. SHARPENING FILTER — enhances edges by subtracting blurred from original
t = time.time()
print("[3] Sharpening Filter 3x3...")
sharpen_kernel = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float64)
img_sharpen = apply_filter_rgb(img_small, sharpen_kernel)
save("sharpen_filter.png", img_sharpen)
print(f"    Time: {time.time()-t:.2f}s")

# 4. SOBEL EDGE DETECTION (horizontal + vertical)
t = time.time()
print("[4] Sobel Edge Filter (X and Y + Magnitude)...")
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)

# Convert to grayscale: Y = 0.2989*R + 0.5870*G + 0.1140*B
gray = np.dot(img_small[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)

edges_x = convolve2d(gray, sobel_x)
edges_y = convolve2d(gray, sobel_y)
edges_magnitude = np.sqrt(edges_x**2 + edges_y**2)
edges_magnitude = (edges_magnitude / edges_magnitude.max() * 255).astype(np.uint8)

save("sobel_x_filter.png", np.stack([np.clip(np.abs(edges_x), 0, 255).astype(np.uint8)] * 3, axis=-1))
save("sobel_y_filter.png", np.stack([np.clip(np.abs(edges_y), 0, 255).astype(np.uint8)] * 3, axis=-1))
save("sobel_magnitude_filter.png", np.stack([edges_magnitude] * 3, axis=-1))
print(f"    Time: {time.time()-t:.2f}s")

# 5. LAPLACIAN FILTER — second derivative, detects edges in all directions
t = time.time()
print("[5] Laplacian Filter 3x3...")
laplacian_kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float64)

laplacian_out = convolve2d(gray, laplacian_kernel)
laplacian_out = np.clip(np.abs(laplacian_out), 0, 255).astype(np.uint8)
save("laplacian_filter.png", np.stack([laplacian_out] * 3, axis=-1))
print(f"    Time: {time.time()-t:.2f}s")

# 6. EMBOSS FILTER — gives 3D shadow/relief effect
t = time.time()
print("[6] Emboss Filter 3x3...")
emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64)
img_emboss = apply_filter_rgb(img_small, emboss_kernel)
save("emboss_filter.png", img_emboss)
print(f"    Time: {time.time()-t:.2f}s")


# ============================================================
# =================== NON-LINEAR FILTERS =====================
# ============================================================
print("\n========== NON-LINEAR FILTERS (rank/order-based) ==========")

# 7. MEDIAN FILTER — replaces pixel with median of neighborhood (removes salt & pepper noise)
t = time.time()
print("[7] Median Filter 5x5...")
img_median = apply_nonlinear_rgb(img_small, 5, np.median)
save("median_filter.png", img_median)
print(f"    Time: {time.time()-t:.2f}s")

# 8. MIN FILTER (EROSION-like) — darkens image, shrinks bright regions
t = time.time()
print("[8] Min Filter 3x3...")
img_min = apply_nonlinear_rgb(img_small, 3, np.min)
save("min_filter.png", img_min)
print(f"    Time: {time.time()-t:.2f}s")

# 9. MAX FILTER (DILATION-like) — brightens image, expands bright regions
t = time.time()
print("[9] Max Filter 3x3...")
img_max = apply_nonlinear_rgb(img_small, 3, np.max)
save("max_filter.png", img_max)
print(f"    Time: {time.time()-t:.2f}s")

# 10. Add salt & pepper noise and show median filter's strength
t = time.time()
print("[10] Salt & Pepper noise + Median denoising demo...")
noisy = img_small.copy()
rng = np.random.default_rng(42)
n_pixels = noisy.shape[0] * noisy.shape[1]
n_salt = int(0.02 * n_pixels)
n_pepper = int(0.02 * n_pixels)

# Salt (white)
salt_y = rng.integers(0, noisy.shape[0], n_salt)
salt_x = rng.integers(0, noisy.shape[1], n_salt)
noisy[salt_y, salt_x] = 255

# Pepper (black)
pepper_y = rng.integers(0, noisy.shape[0], n_pepper)
pepper_x = rng.integers(0, noisy.shape[1], n_pepper)
noisy[pepper_y, pepper_x] = 0

save("noisy_salt_pepper_filter.png", noisy)

img_denoised = apply_nonlinear_rgb(noisy, 5, np.median)
save("denoised_median_filter.png", img_denoised)
print(f"    Time: {time.time()-t:.2f}s")

print("\n All filtered images saved in foundations/ with *_filter.png tag!")

# ============================================================
# ========== BILATERAL FILTER (edge-preserving) ==============
# ============================================================
print("[11] Bilateral Filter 7x7, sigma_color=30, sigma_space=3 (slow)...")
t = time.time()
img_bilateral = bilateral_filter(img_small, diameter=7, sigma_color=30, sigma_space=3)
save("bilateral_filter.png", img_bilateral)
print(f"    Time: {time.time()-t:.2f}s")

# ============================================================
# ========== MORPHOLOGICAL FILTERS (erosion/dilation) ========
# ============================================================
print("[12] Morphological Erosion 5x5...")
t = time.time()
img_erosion = morphological_filter(img_small, size=5, mode='erosion')
save("morphological_erosion_filter.png", img_erosion)
print(f"    Time: {time.time()-t:.2f}s")

print("[13] Morphological Dilation 5x5...")
t = time.time()
img_dilation = morphological_filter(img_small, size=5, mode='dilation')
save("morphological_dilation_filter.png", img_dilation)
print(f"    Time: {time.time()-t:.2f}s")

# ============================================================
# ========== CANNY EDGE DETECTOR (NumPy implementation) ======
# ============================================================
def gaussian_blur(image, size=5, sigma=1.0):
    """Apply Gaussian blur to a grayscale image."""
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image, kernel)

def nonmax_suppression(grad_mag, grad_dir):
    """Non-maximum suppression for edge thinning."""
    h, w = grad_mag.shape
    out = np.zeros_like(grad_mag)
    angle = grad_dir * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = grad_mag[i, j+1]
                r = grad_mag[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = grad_mag[i+1, j-1]
                r = grad_mag[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = grad_mag[i+1, j]
                r = grad_mag[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = grad_mag[i-1, j-1]
                r = grad_mag[i+1, j+1]
            if (grad_mag[i,j] >= q) and (grad_mag[i,j] >= r):
                out[i,j] = grad_mag[i,j]
    return out

def hysteresis(img, low, high):
    """Hysteresis thresholding for edge tracking."""
    strong = 255
    weak = 75
    out = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    out[strong_i, strong_j] = strong
    out[weak_i, weak_j] = weak
    h, w = img.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if out[i,j] == weak:
                if np.any(out[i-1:i+2, j-1:j+2] == strong):
                    out[i,j] = strong
                else:
                    out[i,j] = 0
    return out

def canny_edge_detector(image, low_thresh=50, high_thresh=100):
    """Canny edge detector for grayscale image (NumPy only)."""
    # 1. Gaussian blur
    blur = gaussian_blur(image, size=5, sigma=1.0)
    # 2. Compute gradients (Sobel)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
    gx = convolve2d(blur, sobel_x)
    gy = convolve2d(blur, sobel_y)
    grad_mag = np.hypot(gx, gy)
    grad_mag = grad_mag / grad_mag.max() * 255
    grad_dir = np.arctan2(gy, gx)
    # 3. Non-maximum suppression
    nms = nonmax_suppression(grad_mag, grad_dir)
    # 4. Hysteresis thresholding
    edges = hysteresis(nms, low_thresh, high_thresh)
    return edges.astype(np.uint8)

print("[14] Canny Edge Detector (NumPy, 5x5 blur, thresholds 50/100)...")
t = time.time()
# Use grayscale version of img_small
gray = np.dot(img_small[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
edges_canny = canny_edge_detector(gray, low_thresh=50, high_thresh=100)
save("canny_edge_filter.png", np.stack([edges_canny]*3, axis=-1))
print(f"    Time: {time.time()-t:.2f}s")
