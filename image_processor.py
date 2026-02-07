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
# Output File Tests
# -------------------------

def test_all_effect_files_created():
    """Check that all 6 effects are created for each input image"""
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    expected_effects = ["_posterize", "_anime", "_sepia", "_dream", "_clahe", "_threshold"]

    for input_file in input_files:
        name, ext = os.path.splitext(input_file)

        for effect in expected_effects:
            expected_file = f"{name}{effect}{ext}"
            output_path = os.path.join(OUTPUT_DIR, expected_file)
            assert os.path.exists(output_path), f"Missing output: {expected_file}"

    print("✅ All 6 effects created for each input image")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])