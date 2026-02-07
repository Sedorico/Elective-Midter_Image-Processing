import os
import pytest

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

# List of all effects (updated)
EFFECTS = [
    "_posterize",
    "_anime",
    "_sepia",
    "_dream",
    "_clahe",
    "_threshold"
]

def test_techniques_summary():
    """Check each technique, report counts and missing files, always print summary"""

    # Check folders exist
    assert os.path.exists(INPUT_DIR), f"Input folder '{INPUT_DIR}' does not exist"
    assert os.path.exists(OUTPUT_DIR), f"Output folder '{OUTPUT_DIR}' does not exist"

    # Get input and output images
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(input_files) > 0, "No input images found in input_images folder"

    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(output_files) > 0, "Output folder is empty! No images found"

    # Store results per technique
    technique_results = {}

    for effect in EFFECTS:
        present_count = 0
        missing_files = []
        for input_file in input_files:
            name, ext = os.path.splitext(input_file)
            expected_file = f"{name}{effect}{ext}"
            output_path = os.path.join(OUTPUT_DIR, expected_file)
            if os.path.exists(output_path):
                present_count += 1
            else:
                missing_files.append(expected_file)
        technique_results[effect] = {
            "present": present_count,
            "missing": missing_files
        }

    # -------------------------
    # Print summary for all techniques
    # -------------------------
    print("\n=== Technique Check Summary ===")
    total_passed = 0
    total_failed = 0

    for effect, data in technique_results.items():
        if len(data["missing"]) == 0:
            print(f"✅ {effect[1:]} PASSED ({data['present']}/{len(input_files)} images present)")
            total_passed += 1
        else:
            print(f"❌ {effect[1:]} FAILED ({data['present']}/{len(input_files)} images present, {len(data['missing'])} missing)")
            print(f"   Missing files: {', '.join(data['missing'])}")
            total_failed += 1

    print(f"\nTotal Techniques Passed: {total_passed}/{len(EFFECTS)}, Total Failed: {total_failed}/{len(EFFECTS)}\n")

    # Fail the test if any technique has missing images
    assert total_failed == 0, "Some techniques are missing output images!"
