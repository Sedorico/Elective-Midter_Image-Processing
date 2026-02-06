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

            # Original effects
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

            # NEW: Pencil Sketch
            sketch = pencil_sketch(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_sketch{ext}"), sketch)
            print(f"  → Pencil Sketch saved")

            # NEW: Posterize
            posterized = posterize(img, levels=4)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_posterize{ext}"), posterized)
            print(f"  → Posterize saved")

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