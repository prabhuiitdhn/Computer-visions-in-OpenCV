"""
Image Morphological Operations using NumPy
Base image: BN5.png

Operations implemented:
  1. Erosion
  2. Dilation
  3. Opening  (erosion → dilation)
  4. Closing  (dilation → erosion)
  5. Morphological Gradient (dilation - erosion)
  6. Top Hat  (original - opening)
  7. Black Hat (closing - original)

Structuring elements: square, cross, disk

All outputs saved as *_morph.png in foundations/.
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


save("grayscale_morph.png", gray)


# ============================================================
# STRUCTURING ELEMENTS
# ============================================================
def square_se(size):
    """Square structuring element (all ones)."""
    return np.ones((size, size), dtype=np.uint8)


def cross_se(size):
    """Cross (+) structuring element."""
    se = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    se[mid, :] = 1   # horizontal bar
    se[:, mid] = 1   # vertical bar
    return se


def disk_se(radius):
    """Disk structuring element."""
    size = 2 * radius + 1
    se = np.zeros((size, size), dtype=np.uint8)
    cy, cx = radius, radius
    for y in range(size):
        for x in range(size):
            if (y - cy)**2 + (x - cx)**2 <= radius**2:
                se[y, x] = 1
    return se


# Print structuring elements for reference
print("\n========== STRUCTURING ELEMENTS ==========")
for name, se in [("Square 5x5", square_se(5)),
                 ("Cross 5x5", cross_se(5)),
                 ("Disk r=2 (5x5)", disk_se(2))]:
    print(f"\n{name}:")
    for row in se:
        print("  ", " ".join(str(v) for v in row))


# ============================================================
# CORE MORPHOLOGICAL OPERATIONS (vectorized with stride tricks)
# ============================================================
def morph_op(image, se, func):
    """
    Apply morphological operation on grayscale image.
    - se: structuring element (binary 2D array)
    - func: 'min' for erosion, 'max' for dilation
    """
    kh, kw = se.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad: use max for erosion (so border doesn't shrink),
    #       use min for dilation (so border doesn't grow)
    if func == 'min':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=255)
    else:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode='constant', constant_values=0)

    padded = padded.astype(np.float64)
    h, w = image.shape
    shape = (h, w, kh, kw)
    strides = padded.strides * 2
    windows = as_strided(padded, shape=shape, strides=strides)

    # Mask: only consider pixels where SE is 1
    # Reshape windows to (h, w, kh*kw) and SE to flat mask
    flat_windows = windows.reshape(h, w, -1)
    se_mask = se.ravel().astype(bool)

    # Extract only the pixels under the SE
    masked = flat_windows[:, :, se_mask]

    if func == 'min':
        result = np.min(masked, axis=2)
    else:
        result = np.max(masked, axis=2)

    return np.clip(result, 0, 255).astype(np.uint8)


def erode(image, se):
    """Erosion: output = min over SE neighborhood."""
    return morph_op(image, se, 'min')


def dilate(image, se):
    """Dilation: output = max over SE neighborhood."""
    return morph_op(image, se, 'max')


def opening(image, se):
    """Opening: erosion then dilation (removes small bright noise)."""
    return dilate(erode(image, se), se)


def closing(image, se):
    """Closing: dilation then erosion (fills small dark holes)."""
    return erode(dilate(image, se), se)


def morph_gradient(image, se):
    """Morphological gradient: dilation - erosion (edge extraction)."""
    return (dilate(image, se).astype(np.int16) -
            erode(image, se).astype(np.int16)).clip(0, 255).astype(np.uint8)


def top_hat(image, se):
    """Top hat: original - opening (extracts bright details smaller than SE)."""
    return (image.astype(np.int16) -
            opening(image, se).astype(np.int16)).clip(0, 255).astype(np.uint8)


def black_hat(image, se):
    """Black hat: closing - original (extracts dark details smaller than SE)."""
    return (closing(image, se).astype(np.int16) -
            image.astype(np.int16)).clip(0, 255).astype(np.uint8)


# ============================================================
# APPLY ALL OPERATIONS
# ============================================================
# Use square 5x5 as default SE
se = square_se(5)

print("\n========== MORPHOLOGICAL OPERATIONS (square 5x5 SE) ==========")

# 1. EROSION
t = time.time()
print("[1] Erosion...")
img_erode = erode(gray, se)
save("erosion_morph.png", img_erode)
print(f"    Time: {time.time()-t:.4f}s")

# 2. DILATION
t = time.time()
print("[2] Dilation...")
img_dilate = dilate(gray, se)
save("dilation_morph.png", img_dilate)
print(f"    Time: {time.time()-t:.4f}s")

# 3. OPENING
t = time.time()
print("[3] Opening (erosion → dilation)...")
img_open = opening(gray, se)
save("opening_morph.png", img_open)
print(f"    Time: {time.time()-t:.4f}s")

# 4. CLOSING
t = time.time()
print("[4] Closing (dilation → erosion)...")
img_close = closing(gray, se)
save("closing_morph.png", img_close)
print(f"    Time: {time.time()-t:.4f}s")

# 5. MORPHOLOGICAL GRADIENT
t = time.time()
print("[5] Morphological Gradient (dilation - erosion)...")
img_grad = morph_gradient(gray, se)
save("gradient_morph.png", img_grad)
print(f"    Time: {time.time()-t:.4f}s")

# 6. TOP HAT
t = time.time()
print("[6] Top Hat (original - opening)...")
img_tophat = top_hat(gray, se)
save("tophat_morph.png", img_tophat)
print(f"    Time: {time.time()-t:.4f}s")

# 7. BLACK HAT
t = time.time()
print("[7] Black Hat (closing - original)...")
img_blackhat = black_hat(gray, se)
save("blackhat_morph.png", img_blackhat)
print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# BONUS: Compare structuring element shapes
# ============================================================
print("\n========== SE SHAPE COMPARISON (erosion with different SEs) ==========")

for name, se_item in [("square_5x5", square_se(5)),
                       ("cross_5x5", cross_se(5)),
                       ("disk_r2", disk_se(2))]:
    t = time.time()
    print(f"[SE] Erosion with {name}...")
    result = erode(gray, se_item)
    save(f"erosion_{name}_morph.png", result)
    print(f"    Time: {time.time()-t:.4f}s")


# ============================================================
# BONUS: Noise cleanup demo (opening removes salt, closing removes pepper)
# ============================================================
print("\n========== NOISE CLEANUP DEMO ==========")

# Create binary image using Otsu threshold
from image_thresholding import otsu_threshold
otsu_t, binary = otsu_threshold(gray)
save("binary_morph.png", binary)
print(f"  Otsu threshold = {otsu_t}")

# Add salt noise (random white pixels)
rng = np.random.default_rng(42)
noisy_salt = binary.copy()
salt_mask = rng.random(binary.shape) < 0.02
noisy_salt[salt_mask] = 255
save("noisy_salt_morph.png", noisy_salt)

# Opening removes salt noise
t = time.time()
print("[Demo] Opening removes salt noise...")
cleaned_salt = opening(noisy_salt, square_se(3))
save("cleaned_salt_opening_morph.png", cleaned_salt)
print(f"    Time: {time.time()-t:.4f}s")

# Add pepper noise (random black pixels)
noisy_pepper = binary.copy()
pepper_mask = rng.random(binary.shape) < 0.02
noisy_pepper[pepper_mask] = 0
save("noisy_pepper_morph.png", noisy_pepper)

# Closing fills pepper noise
t = time.time()
print("[Demo] Closing fills pepper noise...")
cleaned_pepper = closing(noisy_pepper, square_se(3))
save("cleaned_pepper_closing_morph.png", cleaned_pepper)
print(f"    Time: {time.time()-t:.4f}s")


print("\n All morphological images saved in foundations/ with *_morph.png tag!")
