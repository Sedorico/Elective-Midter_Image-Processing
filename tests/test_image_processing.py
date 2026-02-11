import os
import cv2
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processor import (
    process_images,
    clahe_process,
    adaptive_threshold_process,
    posterize,
    sepia_process,
    dream_soft_focus,
    anime_effect
)

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"


# -------------------------
# Basic tests
# -------------------------

def test_process_images_runs():
    """Run the processing function"""
    result = process_images()
    assert result is True, "process_images() did not return True"
    print("✅ process_images() ran successfully")


def test_output_directory_exists():
    """Check that the output folder exists"""
    assert os.path.exists(OUTPUT_DIR), f"Output directory '{OUTPUT_DIR}' does not exist"
    print("✅ Output directory exists")


def test_output_images_created():
    """Check that output images are created"""
    files = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    assert len(files) > 0, "No images created in output directory"
    print(f"✅ Number of output images: {len(files)}")


# -------------------------
# Individual Function Tests
# -------------------------

class TestImageFunctions:
    """Test individual image processing functions"""

    def setup_method(self):
        """Create test image before each test"""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_clahe_process(self):
        """Test CLAHE enhancement"""
        result = clahe_process(self.test_image)
        assert result is not None
        assert len(result.shape) == 2
        print("✅ CLAHE function works")

    def test_adaptive_threshold_process(self):
        """Test adaptive threshold"""
        result = adaptive_threshold_process(self.test_image)
        assert result is not None
        print("✅ Adaptive Threshold function works")

    def test_posterize(self):
        """Test posterize effect"""
        result = posterize(self.test_image, levels=4)
        assert result is not None
        assert result.shape == self.test_image.shape
        assert result.dtype == np.uint8
        print("✅ Posterize function works")

    def test_sepia_process(self):
        """Test sepia effect"""
        result = sepia_process(self.test_image)
        assert result is not None
        assert result.shape == self.test_image.shape
        print("✅ Sepia function works")

    def test_dream_soft_focus(self):
        """Test dream soft focus effect"""
        result = dream_soft_focus(self.test_image)
        assert result is not None
        assert result.shape == self.test_image.shape
        print("✅ Dream Soft Focus function works")

    def test_anime_effect(self):
        """Test anime effect"""
        result = anime_effect(self.test_image)
        assert result is not None
        assert result.shape == self.test_image.shape
        print("✅ Anime Effect function works")


# -------------------------
# Technique Summary Test
# -------------------------

EFFECTS = [
    "_posterize",
    "_anime",
    "_sepia",
    "_dream",
    "_clahe",
    "_threshold"
]


def test_techniques_summary():
    """Check each technique, report counts and missing files"""
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
    
    # Print summary
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
    
    # Fail if any technique has missing files
    assert total_failed == 0, "Some techniques are missing output images!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])