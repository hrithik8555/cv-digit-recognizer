# Computer Vision & Digit Recognizer

A pipeline demonstrating fundamental image processing (convolution, filtering) alongside a deep learning approach for recognizing handwritten digits (inspired by MNIST).

## Features
* **Image Processing Fundamentals:** Implementations of convolutional kernels and filters showcasing how features are extracted from images.
* **Digit Recognition:** A fully functional trained model for classifying digits, utilizing modern deep learning architectures.

*(Animations demonstrating kernel passes)*
![2D Convolution Animation](2D_Convolution_Animation.gif)
![Colored Convolution](coloredconv.gif)

## Architecture
- `src/models`: Neural network architectures for image classification.
- `src/utils`: Custom image loading and augmentations.
- `src/scripts`: Scripts to train or evaluate models on raw data.
- `notebooks/`: Step-by-step walkthroughs encompassing Filter fundamentals and Recognition modeling.

## Installation

Ensure you have Python 3.9+ safely installed.

```bash
git clone <repository_url>
cd cv-digit-recognizer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Note: To run the digit recognizer on Kaggle's MNIST dataset, place `train.csv` and `test.csv` inside `data/raw/`. These are ignored by git to save space.
