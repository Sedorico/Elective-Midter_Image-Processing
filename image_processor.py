import cv2
import os
import numpy as np

# ------------------------- Absolute Paths -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------- Effects -------------------------

def clahe_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def adaptive_threshold_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]


def posterize(img, levels=4):
    levels = max(2, min(8, levels))
    step = 256 // levels
    return ((img // step) * step).astype(np.uint8)


def sepia_process(img):
    img_float = img.astype(np.float32)
    sepia_filter = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_img = cv2.transform(img_float, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


def dream_soft_focus(img):
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    return cv2.addWeighted(img, 0.7, blurred, 0.3, 0)


def anime_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


# ------------------------- Main Processing -------------------------

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
    effects = {
        "_posterize": posterize,
        "_anime": anime_effect,
        "_sepia": sepia_process,
        "_dream": dream_soft_focus,
        "_clahe": clahe_process,
        "_threshold": adaptive_threshold_process
    }

    new_files_created = 0

    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠ Could not read: {filename}")
            continue

            clahe_img = clahe_process(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_clahe{ext}"), clahe_img)
            print(f"  → CLAHE saved")
        name, ext = os.path.splitext(filename)
        print(f"\n✓ Processing: {filename}")

        for suffix, func in effects.items():
            output_file = os.path.join(OUTPUT_DIR, f"{name}{suffix}{ext}")

            if os.path.exists(output_file):
                print(f"  → {suffix[1:].capitalize()} already exists, skipped")
                continue

            if suffix == "_posterize":
                result = func(img, levels=4)
            else:
                result = func(img)

            sketch = pencil_sketch(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_sketch{ext}"), sketch)
            print(f"  → Pencil Sketch saved")

            posterized = posterize(img, levels=4)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_posterize{ext}"), posterized)
            print(f"  → Posterize saved")
            cv2.imwrite(output_file, result)
            print(f"  → {suffix[1:].capitalize()} saved at {output_file}")
            new_files_created += 1

    if new_files_created == 0:
        print("\n⚠ No new images created, all outputs already exist")
    else:
        print(f"\n✓ Total new images created: {new_files_created}")

            # NEW EFFECTS
            neon = neon_border_effect(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_neon{ext}"), neon)
            print(f"  → Neon Border saved")

            dreamy = dreamy_soft_focus_effect(img)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_dreamy{ext}"), dreamy)
            print(f"  → Dreamy Soft Focus saved")

    return True


# ------------------------- Run -------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("IMAGE PROCESSING STARTED")
    print("=" * 50)
    process_images()
    print("=" * 50)
    print("✓ Image processing completed!")
    print(f"✓ Check '{OUTPUT_DIR}' folder for results")
    print("=" * 50)
