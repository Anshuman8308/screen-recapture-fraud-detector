
# Screen Recapture Fraud Detector

A lightweight computer vision system that detects whether an input image is:

- A genuine camera captured photo (`REAL`)
- A recaptured image photographed from another screen such as a phone, laptop, or monitor (`FAKE`)

This project was built as a fraud detection solution for identifying screen recapture attempts in mobile verification workflows.

---

## Problem Statement

In many mobile applications, users may attempt to bypass verification by photographing another screen displaying an image instead of capturing a real-world object.

The goal is to detect whether an image is:

- Directly captured from a camera
- Re-photographed from a digital display (screen recapture fraud)

The solution should remain lightweight, fast, and suitable for future mobile deployment.

---

## Approach

Instead of using a heavy deep learning model, this system uses handcrafted computer vision features designed specifically to capture screen recapture artifacts.

Extracted features include:

- FFT Frequency Analysis
- Patch-wise Frequency Analysis
- Local Binary Pattern (LBP) Texture Features
- HSV Color Distribution Statistics
- Brightness and Exposure Analysis
- Screen Glare Detection
- Edge Density Features
- Noise and Compression Artifact Analysis

These features are combined into a feature vector and passed to a lightweight ensemble classifier.

---

## Model Used

After training and evaluation of multiple models:

- Random Forest
- Gradient Boosting
- Extra Trees Classifier

Final selected model:

```text
ExtraTreesClassifier
```

Reason:

- Best validation accuracy
- Fast inference
- Lightweight CPU execution

---

## Performance

Dataset collected manually:

```text
50 REAL images
50 SCREEN RECAPTURE images
```

Evaluation results:

```text
Held-out Test Accuracy: >95%
5-Fold Cross Validation Accuracy: 93.6%
```

Inference latency:

```text
~120–150 ms per image (local CPU)
```

---

## Project Structure

```text
.
├── train.py
├── predict.py
├── app.py
├── model.pkl
├── scaler.pkl
├── feature_names.pkl
├── requirements.txt/

```

---

## Installation

Clone repository:

```bash
git clone https://github.com/Anshuman8308/screen-recapture-fraud-detector.git
cd screen-recapture-fraud-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Prediction (CLI)

Run prediction on a single image:

```bash
python predict.py path_to_image.jpg
```

Example:

```bash
python predict.py sample.jpg
```

Output:

```text
Prediction: FAKE
Confidence: 98.42%
```

---

## Running Web Interface

Start Gradio app locally:

```bash
python app.py
```

This launches a browser interface where images can be uploaded for real-time prediction.

---

## Live Demo

[Launch Live Demo](https://anshuman83-screen-recapture-photo-fraud-detector.hf.space)


## Possible Improvements

With a larger production-scale dataset, future improvements could include:

- More screen types (OLED, LCD, AMOLED)
- Printed photo fraud detection
- Adversarial fraud cases
- Mobile optimization using ONNX / TensorFlow Lite
- Threshold calibration for production fraud scoring

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-learn
- Gradio
