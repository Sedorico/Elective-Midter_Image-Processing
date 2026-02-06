import os
import cv2
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processor import (
    process_images,
    clahe_process,
    gaussian_blur_process,
    adaptive_threshold_process,
    invert_colors,
    pencil_sketch,
    posterize
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
        self.test_gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)

    def test_clahe_process(self):
        """Test CLAHE enhancement"""
        result = clahe_process(self.test_image)
        assert result is not None
        assert len(result.shape) == 2
        print("✅ CLAHE function works")

    def test_gaussian_blur_process(self):
        """Test Gaussian blur"""
        result = gaussian_blur_process(self.test_image)
        assert result is not None
        assert len(result.shape) == 2
        print("✅ Gaussian Blur function works")

    def test_adaptive_threshold_process(self):
        """Test adaptive threshold"""
        result = adaptive_threshold_process(self.test_image)
        assert result is not None
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        print("✅ Adaptive Threshold function works")

    def test_invert_colors(self):
        """Test color inversion"""
        result = invert_colors(self.test_gray)
        assert result is not None
        assert np.array_equal(result, 255 - self.test_gray)
        print("✅ Invert Colors function works")

    def test_pencil_sketch(self):
        """Test pencil sketch effect"""
        result = pencil_sketch(self.test_image)
        assert result is not None
        assert len(result.shape) == 2
        assert result.dtype == np.uint8
        print("✅ Pencil Sketch function works")

    def test_posterize(self):
        """Test posterize effect"""
        result = posterize(self.test_image, levels=4)
        assert result is not None
        assert result.shape == self.test_image.shape
        assert result.dtype == np.uint8
        print("✅ Posterize function works")


# -------------------------
# Output File Tests
# -------------------------

def test_all_effect_files_created():
    """Check that all 6 effects are created for each input image"""
    input_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for input_file in input_files:
        name, ext = os.path.splitext(input_file)

        expected_outputs = [
            f"{name}_clahe{ext}",
            f"{name}_blurred{ext}",
            f"{name}_threshold{ext}",
            f"{name}_inverted{ext}",
            f"{name}_sketch{ext}",
            f"{name}_posterize{ext}"
        ]

        for expected_file in expected_outputs:
            output_path = os.path.join(OUTPUT_DIR, expected_file)
            assert os.path.exists(output_path), f"Missing output: {expected_file}"

    print("✅ All 6 effects created for each input image")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])