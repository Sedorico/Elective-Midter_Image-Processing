import cv2
import os
import numpy as np

# ------------------------- Directories -------------------------
# Set the paths for the input and output folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current file's path
INPUT_DIR = os.path.join(BASE_DIR, "input_images")     # Input folder with the original images
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")   # Output folder for processed images

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the output folder if it doesn't exist

# ------------------------- Effects -------------------------

# Apply CLAHE (improves image contrast)
def clahe_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    return clahe.apply(gray)  # Apply CLAHE to the image

# Convert the image to black and white using thresholding
def adaptive_threshold_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]  # Apply thresholding

# Reduce the number of colors in the image (posterize effect)
def posterize(img, levels=4):
    levels = max(2, min(8, levels))  # Ensure levels are between 2 and 8
    step = 256 // levels  # Calculate step size for colors
    return ((img // step) * step).astype(np.uint8)  # Apply posterize effect

# Apply a sepia (vintage) look to the image
def sepia_process(img):
    img_float = img.astype(np.float32)  # Convert to float for better precision
    sepia_filter = np.array([  # Sepia filter for color change
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])
    sepia_img = cv2.transform(img_float, sepia_filter)  # Apply the sepia filter
    return np.clip(sepia_img, 0, 255).astype(np.uint8)  # Ensure valid image color range

# Apply a dreamy, soft-focus effect for a smooth look
def dream_soft_focus(img):
    blurred = cv2.GaussianBlur(img, (21, 21), 0)  # Apply a Gaussian blur
    return cv2.addWeighted(img, 0.7, blurred, 0.3, 0)  # Blend the blurred and original images

# Apply realistic anime style using edges, smoothing, and color quantization
def anime_style_realistic(img, k=8):
    """
    Makes the image look like anime by applying edge detection, 
    smoothing, and reducing the number of colors.
    """

    # Step 1: Smooth the image for a flat color look (cel-shading effect)
    img_color = img.copy()
    for _ in range(2):  # Apply bilateral filter multiple times for better smoothing
        img_color = cv2.bilateralFilter(img_color, 9, 75, 75)

    # Step 2: Detect edges in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray, 100, 200)  # Use Canny edge detection
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert edges back to 3-channel
    edges = cv2.bitwise_not(edges)  # Invert edges to make them white on black

    # Step 3: Reduce the number of colors in the image (anime look)
    data = np.float32(img_color).reshape((-1, 3))  # Flatten image for clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # K-means clustering
    center = np.uint8(center)  # Convert color centers back to uint8
    result = center[label.flatten()]  # Replace each pixel with the nearest center
    result = result.reshape(img_color.shape)  # Reshape back to original image shape

    # Step 4: Make edges thicker for a stronger outline
    enhanced_edges = cv2.dilate(edges, None, iterations=2)  # Make edges thicker
    enhanced_edges = cv2.cvtColor(enhanced_edges, cv2.COLOR_BGR2GRAY)  # Convert back to grayscale
    enhanced_edges = cv2.cvtColor(enhanced_edges, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel

    # Step 5: Combine the smoothed colors and enhanced edges
    anime_image = cv2.bitwise_and(result, enhanced_edges)  # Combine smoothed color and edges

    # Step 6: Merge with original edges for final anime effect
    final_output = cv2.bitwise_or(anime_image, enhanced_edges)  # Final combine

    return final_output


# ------------------------- New Effect: Mirror -------------------------
# Create a mirrored version of the image by flipping it
def mirror_effect(img):
    """Creates a mirrored version of the image"""
    mirrored_img = cv2.flip(img, 1)  # Flip the image horizontally
    mirrored_image = np.concatenate((img, mirrored_img), axis=1)  # Combine original and mirrored images
    return mirrored_image  # Return the mirrored image


# ------------------------- Main Processing -------------------------
def process_images():
    effects = {
        "_posterize": posterize,      # Apply posterize effect
        "_anime": anime_style_realistic,  # Apply realistic anime effect
        "_sepia": sepia_process,      # Apply sepia effect
        "_dream": dream_soft_focus,   # Apply dreamy soft focus effect
        "_clahe": clahe_process,      # Apply CLAHE effect
        "_threshold": adaptive_threshold_process,  # Apply adaptive threshold effect
        "_mirror": mirror_effect      # Apply mirror effect
    }

    new_files_created = 0  # Counter for new files created

    # Process each file in the input folder
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Process only image files
            continue

        img_path = os.path.join(INPUT_DIR, filename)  # Get the image path
        img = cv2.imread(img_path)  # Read the image

        if img is None:  # Skip if the image is not valid
            print(f"⚠ Could not read: {filename}")
            continue

        name, ext = os.path.splitext(filename)  # Get the file name and extension
        print(f"\n✓ Processing: {filename}")

        # Apply each effect and save the output
        for suffix, func in effects.items():
            output_file = os.path.join(OUTPUT_DIR, f"{name}{suffix}{ext}")  # Output file path

            if os.path.exists(output_file):  # Skip if output file already exists
                print(f"  → {suffix[1:].capitalize()} already exists, skipped")
                continue

            if suffix == "_posterize":  # Special handling for posterize (it has `levels`)
                result = func(img, levels=4)
            else:
                result = func(img)  # Apply the effect

            cv2.imwrite(output_file, result)  # Save the processed image
            print(f"  → {suffix[1:].capitalize()} saved at {output_file}")
            new_files_created += 1  # Increment counter for new files created

    # Summary of results
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
    process_images()  # Start the image processing
    print("=" * 50)
    print("✓ Image processing completed!")
    print(f"✓ Check '{OUTPUT_DIR}' folder for results")  # Final message
    print("=" * 50)
