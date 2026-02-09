# Image Processing Pipeline with CI/CD

Automated image processing application with multiple filter effects using Python, OpenCV, and GitHub Actions.

## 🎨 Image Processing Effects

1. **Posterize** - Reduces colors for a pop-art style
2. **Anime** - Transforms images into an anime-style appearance
3. **Sepia Effect** - Applies a warm, vintage tone
4. **Dream Soft Focus** - Creates a soft, glowing dream-like look
5. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** - Enhances image contrast
6. **Adaptive Threshold** - Converts images into binary format using local adaptation

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
   - `filename_posterize.jpg` - Posterize effect
   - `filename_anime.jpg` - Anime-style effect
   - `filename_sepia.jpg` - Sepia effect
   - `filename_dream_soft_focus.jpg` - Dream soft focus effect
   - `filename_clahe.jpg` - CLAHE enhanced
   - `filename_threshold.jpg` - Adaptive threshold
     

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

- [KARL CEDRIC DEL CARMEN] - [DEV OPS ENGINEER]
- [CARLO ESTACIO] - [DOCUMENTER/PRESENTER]
- [JHAN REY MAHSAKAY] - [IMAGE PROCESSING LEAD]
- [SEAN WENDEL VILLAMAYOR] - [TESTER]

---

**Built with ❤️ for Image Processing Midterm Project**
