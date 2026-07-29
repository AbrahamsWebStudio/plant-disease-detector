# 🌿 Plant Disease Detector

A machine learning-powered web application that detects diseases in tomato plant leaves from uploaded images. The project combines a deep learning image classification model with a Django web interface, allowing users to upload a leaf image and receive a predicted disease class along with a confidence score.

---

## Project Overview

Plant diseases significantly reduce agricultural productivity. This project demonstrates how machine learning can assist in the early detection of tomato leaf diseases using image classification.

The application allows users to:

* Upload an image of a tomato leaf
* Predict the disease affecting the leaf
* Display the predicted class
* Display the model's confidence score

---

## Supported Disease Classes

The model currently classifies tomato leaves into the following categories:

* 🍃 Tomato Healthy
* 🟤 Tomato Early Blight
* ⚫ Tomato Late Blight

---

## Features

* Deep learning-based image classification
* Django web application
* Image upload interface
* Prediction confidence score
* Modular training pipeline
* Unit tests for model components
* Easy-to-follow project structure

---

# Project Structure

```text
plant_disease_detector/
│
├── dataset/                     # Training dataset (not included)
│   ├── Tomato_Early_Blight/
│   ├── Tomato_Healthy/
│   └── Tomato_Late_Blight/
│
├── model/
│   ├── predictor.py
│   ├── train_pipeline.py
│   ├── tests/
│   └── README.md
│
├── web/
│   ├── classifier/
│   ├── manage.py
│   └── plant_disease_detector/
│
├── requirements.txt
├── README.md
└── TESTING_GUIDE.md
```

---

# Technologies Used

* Python
* Django
* TensorFlow / Keras
* NumPy
* Pillow
* Pytest

---

# Requirements

* Python 3.12 or later
* pip
* Git

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/AbrahamsWebStudio/plant-disease-detector.git
cd plant-disease-detector
```

---

## 2. Create a virtual environment

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The dataset is **not included** in this repository because of its size.

Download it from:

https://drive.google.com/file/d/1yosGXdRSwgZ0kE8faAF-9QFADFATAJIB/view?usp=sharing

OR

  https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease

After downloading, extract it so the project structure becomes:

```text
plant_disease_detector/
│
├── dataset/
│   ├── Tomato_Early_Blight/
│   ├── Tomato_Healthy/
│   └── Tomato_Late_Blight/
```

---

# Training the Model

Run:

```bash
python model/train_pipeline.py
```

If the dataset has been correctly installed, the training process will begin automatically.

---

# Running the Web Application

Navigate to the Django project:

```bash
cd web
```

Start the development server:

```bash
python manage.py runserver
```

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

---

# Using the Application

1. Open the home page.
2. Click **Choose File**.
3. Select a tomato leaf image.
4. Click **Predict**.
5. View the predicted disease class and confidence score.

---

# Running Tests

Run the unit tests using:

```bash
pytest
```

or

```bash
pytest model/tests
```

---

# Testing Guide

Detailed testing procedures are available in:

```text
TESTING_GUIDE.md
```

---

# Expected Output

When the application is running correctly:

* The home page loads successfully.
* Users can upload a tomato leaf image.
* The model predicts one of the supported disease classes.
* The confidence score is displayed.
* Invalid uploads are handled gracefully.

---

# Troubleshooting

## ModuleNotFoundError

Activate the virtual environment and reinstall dependencies.

```bash
pip install -r requirements.txt
```

---

## Dataset Not Found

Verify that the dataset folder exists:

```text
dataset/
├── Tomato_Early_Blight/
├── Tomato_Healthy/
└── Tomato_Late_Blight/
```

---

## Model File Missing

Train the model first:

```bash
python model/train_pipeline.py
```

---

## Django Server Will Not Start

Run:

```bash
python manage.py check
```

to identify configuration issues.

---

# Future Improvements

* Support additional plant species
* Detect more disease classes
* Improve prediction accuracy
* Deploy the application to the cloud
* Add user authentication
* Store prediction history
* Mobile-friendly interface

---

# Author

**Abraham Wesley Ochieng**

GitHub: https://github.com/AbrahamsWebStudio

---

# License

This project is intended for educational and research purposes.
