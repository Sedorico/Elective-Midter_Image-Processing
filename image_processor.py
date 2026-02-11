import cv2
import os
import numpy as np

# ------------------------- Absolute Paths -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------- Effects -------------------------

# Existing effects ...

def retro_filter(img):
    # Convert to a more vintage look
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for color manipulation
    img = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 50)  # Add a warm tint to the image

    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Apply vignette effect
    rows, cols = img.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 5)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 5)
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    img = cv2.filter2D(img, -1, mask)

    return img


def mirror_effect(img):
    # Create the mirror effect by flipping the image horizontally
    mirrored_img = cv2.flip(img, 1)

    # Concatenate the original image and mirrored image side by side
    mirrored_image = np.concatenate((img, mirrored_img), axis=1)
    return mirrored_image


# ------------------------- Main Processing -------------------------

def process_images():
    effects = {
        "_posterize": posterize,
        "_anime": anime_effect,
        "_sepia": sepia_process,
        "_dream": dream_soft_focus,
        "_clahe": clahe_process,
        "_threshold": adaptive_threshold_process,
        "_retro": retro_filter,
        "_mirror": mirror_effect
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
