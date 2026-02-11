# Image Processing Pipeline with CI/CD

Automated image processing application with multiple filter effects using Python, OpenCV, and GitHub Actions.

## 🎨 Image Processing Effects

### Effects:
1. **Posterize** – Reduces number of colors to create a flat, poster-like effect
2. **Anime Effect** – Stylizes image to look like anime/cartoon art
3. **Sepia Effect** – Applies warm brown tone for vintage look
4. **Dream Soft Focus** – Adds soft blur for dreamy appearance
5. **CLAHE** – Enhances image contrast and details
6. **Threshold** – Converts image to black and white based on intensity
7. **Mirror Effect** – Flips image horizontally to create reflection


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

   * `filename_posterize.jpg` - Posterize effect
   * `filename_anime.jpg` - Anime style effect
   * `filename_sepia.jpg` - Sepia vintage effect
   * `filename_dreamsoft.jpg` - Dream soft focus effect
   * `filename_clahe.jpg` - CLAHE contrast enhancement
   * `filename_threshold.jpg` - Black & white threshold
   * `filename_mirror.jpg` - Mirror (horizontal flip) effect


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
