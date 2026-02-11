import os
import pytest

# ------------------------- Input and Output Directories -------------------------
# Define the paths for the input and output directories
INPUT_DIR = "input_images"  # The directory where the original images are stored
OUTPUT_DIR = "output_images"  # The directory where the processed images will be saved

# ------------------------- List of Effects -------------------------
# List of image processing effects to check, including the new "mirror" effect
EFFECTS = [
    "_posterize",  # Effect to posterize the image
    "_anime",      # Effect to create an anime/cartoon-like image
    "_sepia",      # Effect to apply a sepia tone for a vintage look
    "_dream",      # Effect for dreamy, soft focus
    "_clahe",      # Effect for contrast limited adaptive histogram equalization
    "_threshold",  # Effect for binary thresholding
    "_mirror"      # Newly added: Mirror effect for creating a mirrored image
]

# ------------------------- Test Function -------------------------
def test_techniques_summary():
    """Check each technique, report counts and missing files, always print summary"""

    # Check if input and output directories exist
    assert os.path.exists(INPUT_DIR), f"Input folder '{INPUT_DIR}' does not exist"
    assert os.path.exists(OUTPUT_DIR), f"Output folder '{OUTPUT_DIR}' does not exist"

    # Get the list of input image files (supports .png, .jpg, .jpeg)
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(input_files) > 0, "No input images found in input_images folder"

    # Get the list of output image files (supports .png, .jpg, .jpeg)
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(output_files) > 0, "Output folder is empty! No images found"

    # Store results for each effect (whether images are processed correctly or missing)
    technique_results = {}

    # Loop through each effect and check if corresponding output images exist
    for effect in EFFECTS:
        present_count = 0
        missing_files = []

        # For each input image, check if the corresponding output file exists
        for input_file in input_files:
            name, ext = os.path.splitext(input_file)  # Get the file name and extension
            expected_file = f"{name}{effect}{ext}"  # Expected output filename based on the effect
            output_path = os.path.join(OUTPUT_DIR, expected_file)  # Full path to the output image

            # If the output file exists, increment present_count
            if os.path.exists(output_path):
                present_count += 1
            else:
                missing_files.append(expected_file)  # If missing, add to missing_files list

        # Save the results for this effect (how many files are present and missing)
        technique_results[effect] = {
            "present": present_count,
            "missing": missing_files
        }

    # -------------------------
    # Print summary of the results for all techniques
    # -------------------------
    print("\n=== Technique Check Summary ===")
    total_passed = 0
    total_failed = 0

    # Loop through each effect and print whether it passed or failed
    for effect, data in technique_results.items():
        if len(data["missing"]) == 0:
            print(f"✅ {effect[1:]} PASSED ({data['present']}/{len(input_files)} images present)")  # No missing files, effect passed
            total_passed += 1
        else:
            print(
                f"❌ {effect[1:]} FAILED ({data['present']}/{len(input_files)} images present, {len(data['missing'])} missing)")  # Some files are missing, effect failed
            print(f"   Missing files: {', '.join(data['missing'])}")  # Print the list of missing files
            total_failed += 1

    # -------------------------
    # Final Summary
    # -------------------------
    print(f"\nTotal Techniques Passed: {total_passed}/{len(EFFECTS)}, Total Failed: {total_failed}/{len(EFFECTS)}\n")

    # Fail the test if any technique has missing output images
    assert total_failed == 0, "Some techniques are missing output images!"
