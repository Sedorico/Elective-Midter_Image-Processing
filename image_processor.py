import cv2
import os
import numpy as np

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"


# ------------------------- Cleanup -------------------------

def cleanup_output_directory():
    """Delete all files in output directory before processing"""
    if os.path.exists(OUTPUT_DIR):
        # Delete everything inside except .gitkeep
        for filename in os.listdir(OUTPUT_DIR):
            if filename == '.gitkeep':
                continue  # Keep .gitkeep file

            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Delete file
                    print(f"  🗑️ Deleted: {filename}")
            except Exception as e:
                print(f'❌ Failed to delete {file_path}. Reason: {e}')

    # Recreate directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("✅ Output directory cleaned\n")


# ------------------------- Effects -------------------------

# CLAHE
def clahe_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# Threshold
def adaptive_threshold_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]


# Posterize
def posterize(img, levels=4):
    levels = max(2, min(8, levels))
    step = 256 // levels
    return ((img // step) * step).astype(np.uint8)


# Sepia Effect
def sepia_process(img):
    img_float = img.astype(np.float32)
    sepia_filter = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_img = cv2.transform(img_float, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


# Dream Soft Focus (blur + glow)
def dream_soft_focus(img):
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    return cv2.addWeighted(img, 0.7, blurred, 0.3, 0)


# Anime Effect (cartoon style)
def anime_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


# ------------------------- Main Processing -------------------------

def process_images():
    # CLEANUP FIRST - Delete old outputs!
    cleanup_output_directory()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    effects = {
        "_posterize": posterize,
        "_anime": anime_effect,
        "_sepia": sepia_process,
        "_dream": dream_soft_focus,
        "_clahe": clahe_process,
        "_threshold": adaptive_threshold_process
    }

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

        # Process each effect
        for suffix, func in effects.items():
            output_file = os.path.join(OUTPUT_DIR, f"{name}{suffix}{ext}")

            if suffix in ["_posterize"]:
                result = func(img, levels=4)
            else:
                result = func(img)

            cv2.imwrite(output_file, result)
            print(f"  → {suffix[1:].capitalize()} saved")

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