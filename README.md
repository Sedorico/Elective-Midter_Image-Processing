# Image Processing Pipeline with CI/CD

Automated image processing application with multiple filter effects using Python, OpenCV, and GitHub Actions.

## 🎨 Image Processing Effects

### Original Effects:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** - Enhances image contrast
2. **Gaussian Blur** - Applies blur effect
3. **Adaptive Threshold** - Binary threshold with local adaptation
4. **Invert Colors** - Inverts image colors

### New Effects:
5. **Pencil Sketch** - Artistic pencil drawing effect
6. **Posterize** - Reduces colors for pop-art style

## 🛠️ Technologies Used

- Python 3.10
- OpenCV (cv2)
- NumPy
- PyTest
- GitHub Actions (CI/CD)

## 📁 Project Structure
```
random/
├── input_images/           # Place images here for processing
├── output_images/          # Processed images saved here
├── tests/
│   └── test_image_processing.py
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions configuration
├── image_processor.py      # Main processing script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
   git clone <your-repo-url>
   cd random
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

## 💻 Usage

1. **Add images** to `input_images/` folder
   - Supported: `.jpg`, `.jpeg`, `.png`

2. **Run the processor**
```bash
   python image_processor.py
```

3. **Check results** in `output_images/` folder:
   - `filename_clahe.jpg` - CLAHE enhanced
   - `filename_blurred.jpg` - Gaussian blur
   - `filename_threshold.jpg` - Adaptive threshold
   - `filename_inverted.jpg` - Inverted colors
   - `filename_sketch.jpg` - Pencil sketch
   - `filename_posterize.jpg` - Posterized

## 🔄 GitHub Actions CI/CD

The pipeline automatically runs on every push:

1. ✅ Installs dependencies
2. ✅ Processes images
3. ✅ Uploads processed images as artifacts
4. ✅ Displays results

## 🎓 Academic Information

**Course**: Image Processing Elective  
**Project**: Midterm Project - CI/CD Pipeline  
**Date**: February 2026

## 👥 Group Members

- [Member 1 Name] - [Role]
- [Member 2 Name] - [Role]
- [Member 3 Name] - [Role]
- [Member 4 Name] - [Role]

---

**Built with ❤️ for Image Processing Midterm Project**