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

            cv2.imwrite(output_file, result)
            print(f"  → {suffix[1:].capitalize()} saved at {output_file}")
            new_files_created += 1

    if new_files_created == 0:
        print("\n⚠ No new images created, all outputs already exist")
    else:
        print(f"\n✓ Total new images created: {new_files_created}")

    return new_files_created

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
