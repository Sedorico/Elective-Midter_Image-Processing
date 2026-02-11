import cv2
import os
import numpy as np

# ------------------------- Absolute Paths -------------------------
# Define base directory and create input and output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #  kunin yung current na path
INPUT_DIR = os.path.join(BASE_DIR, "input_images")     #  input folder kung saan nakalagay yung fresh images
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")   # Output folder dito na save yung na lagyan ng techniques

os.makedirs(OUTPUT_DIR, exist_ok=True)  # gagawa output directory if hindi pa nagagawa

# ------------------------- Effects -------------------------

# CLAHE (Contrast Limited Adaptive Histogram Equalization) for better image contrast
def clahe_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    return clahe.apply(gray)  # Apply CLAHE to the grayscale image

# Adaptive Thresholding for binary image creation (black & white)
def adaptive_threshold_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]  # Apply adaptive thresholding

# Posterize effect (reducing the number of color levels)
def posterize(img, levels=4):
    levels = max(2, min(8, levels))  # Ensure levels are between 2 and 8
    step = 256 // levels  # Step size for color reduction
    return ((img // step) * step).astype(np.uint8)  # Posterize by dividing colors into levels

# Sepia effect (a warm, vintage look applied to the image)
def sepia_process(img):
    img_float = img.astype(np.float32)  # Convert to float for better precision
    sepia_filter = np.array([  # Sepia filter matrix for color transformation
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_img = cv2.transform(img_float, sepia_filter)  # Apply the sepia filter to the image
    return np.clip(sepia_img, 0, 255).astype(np.uint8)  # Clip the values to valid image range (0-255)

# Dreamy soft-focus effect for a smooth, ethereal look
def dream_soft_focus(img):
    blurred = cv2.GaussianBlur(img, (21, 21), 0)  # Apply Gaussian blur to the image
    return cv2.addWeighted(img, 0.7, blurred, 0.3, 0)  # Blend the original image and blurred image

# Anime-style effect using edge detection and bilateral filtering
def anime_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    blur = cv2.medianBlur(gray, 5)  # Apply a median blur to smooth the image
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )  # Create edges using adaptive thresholding
    color = cv2.bilateralFilter(img, 9, 250, 250)  # Apply bilateral filter for edge-preserving smoothing
    return cv2.bitwise_and(color, color, mask=edges)  # Combine edges and smoothed image for cartoon-like effect


# ------------------------- New Effect: Mirror -------------------------
# Creates a mirrored version of the image by flipping it horizontally
def mirror_effect(img):
    """Creates a mirrored version of the image"""
    mirrored_img = cv2.flip(img, 1)  # Flip the image horizontally (left-to-right)
    mirrored_image = np.concatenate((img, mirrored_img), axis=1)  # Concatenate original and mirrored images side by side
    return mirrored_image  # Return the mirrored effect


# ------------------------- Main Processing -------------------------
def process_images():
    effects = {
        "_posterize": posterize,      # Apply posterize effect
        "_anime": anime_effect,       # Apply anime effect
        "_sepia": sepia_process,      # Apply sepia effect
        "_dream": dream_soft_focus,   # Apply dreamy soft focus effect
        "_clahe": clahe_process,      # Apply CLAHE effect
        "_threshold": adaptive_threshold_process,  # Apply adaptive threshold effect
        "_mirror": mirror_effect      # Apply mirror effect
    }

    new_files_created = 0  # Counter for new files created

    # Loop over all files in the input directory
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Only process image files
            continue

        img_path = os.path.join(INPUT_DIR, filename)  # Full path of the input image
        img = cv2.imread(img_path)  # Read the image

        if img is None:  # If the image is not valid, skip it
            print(f"⚠ Could not read: {filename}")
            continue

        name, ext = os.path.splitext(filename)  # Get the file name and extension
        print(f"\n✓ Processing: {filename}")

        # Apply each effect in the effects dictionary
        for suffix, func in effects.items():
            output_file = os.path.join(OUTPUT_DIR, f"{name}{suffix}{ext}")  # Define the output file path

            if os.path.exists(output_file):  # Skip if the output file already exists
                print(f"  → {suffix[1:].capitalize()} already exists, skipped")
                continue

            if suffix == "_posterize":  # Special handling for posterize (it has a `levels` argument)
                result = func(img, levels=4)
            else:
                result = func(img)  # Apply the effect

            cv2.imwrite(output_file, result)  # Save the processed image
            print(f"  → {suffix[1:].capitalize()} saved at {output_file}")
            new_files_created += 1  # Increment counter for new files created

    # Summary of the results
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
    process_images()  # Call the process_images function to start the processing
    print("=" * 50)
    print("✓ Image processing completed!")
    print(f"✓ Check '{OUTPUT_DIR}' folder for results")  # Final message indicating where the results are saved
    print("=" * 50)
