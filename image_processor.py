import cv2
import os
import numpy as np

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"


def clahe_process(img):
    """Apply CLAHE enhancement"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return clahe_img


def gaussian_blur_process(img):
    """Apply Gaussian Blur"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def adaptive_threshold_process(img):
    """Apply Adaptive Threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def invert_colors(img):
    """Invert image colors"""
    return cv2.bitwise_not(img)


def pencil_sketch(img):
    """Apply pencil sketch effect"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return sketch


def posterize(img, levels=4):
    """Apply posterize effect to reduce colors"""
    levels = max(2, min(8, levels))
    step = 256 // levels
    posterized = (img // step) * step
    return posterized.astype(np.uint8)


# ============================================
# NEW EFFECT #1: NEON BORDER
# ============================================
def neon_border_effect(img):
    """Neon glowing border effect"""
    img_copy = img.copy()
    
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    neon_cyan = np.zeros_like(img_copy)
    neon_cyan[:, :, 0] = 255
    neon_cyan[:, :, 1] = 255

    neon_magenta = np.zeros_like(img_copy)
    neon_magenta[:, :, 0] = 255
    neon_magenta[:, :, 2] = 255

    blur_cyan = cv2.GaussianBlur(neon_cyan, (15, 15), 0)
    blur_magenta = cv2.GaussianBlur(neon_magenta, (15, 15), 0)

    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_mask = edges_3channel.astype(float) / 255.0

    result = img_copy.astype(float)
    result += blur_cyan.astype(float) * edges_mask * 0.7
    result += blur_magenta.astype(float) * edges_mask * 0.5

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================
# NEW EFFECT #2: DREAMY SOFT FOCUS
# ============================================
def dreamy_soft_focus_effect(img):
    """Dreamy soft focus effect"""
    img_copy = img.copy()

    soft = cv2.GaussianBlur(img_copy, (35, 35), 0)
    soft = cv2.convertScaleAbs(soft, alpha=1.1, beta=20)

    glow = cv2.GaussianBlur(soft, (51, 51), 0)

    result = cv2.addWeighted(img_copy, 0.5, soft, 0.3, 0)
    result = cv2.addWeighted(result, 0.8, glow, 0.2, 0)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:, :, 1] *= 1.3
    hsv[:, :, 2] *= 1.1

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def process_images():
    """Process all images with multiple effects"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(INPUT_DIR, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠ Could not read: {filename}")
                continue

            name, ext = os.path.splitext(filename)

            print(f"\n✓ Processing: {filename}")

            clahe_img = clahe_process(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_clahe{ext}"), clahe_img)
            print(f"  → CLAHE saved")

            blurred = gaussian_blur_process(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_blurred{ext}"), blurred)
            print(f"  → Gaussian Blur saved")

            thresh = adaptive_threshold_process(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_threshold{ext}"), thresh)
            print(f"  → Adaptive Threshold saved")

            inverted = invert_colors(thresh)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_inverted{ext}"), inverted)
            print(f"  → Inverted saved")

            sketch = pencil_sketch(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_sketch{ext}"), sketch)
            print(f"  → Pencil Sketch saved")

            posterized = posterize(img, levels=4)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_posterize{ext}"), posterized)
            print(f"  → Posterize saved")

            # NEW EFFECTS
            neon = neon_border_effect(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_neon{ext}"), neon)
            print(f"  → Neon Border saved")

            dreamy = dreamy_soft_focus_effect(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_dreamy{ext}"), dreamy)
            print(f"  → Dreamy Soft Focus saved")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("IMAGE PROCESSING STARTED")
    print("=" * 50)
    process_images()
    print("=" * 50)
    print("✓ Image processing completed!")
    print(f"✓ Check '{OUTPUT_DIR}' folder for results")
    print("=" * 50)
