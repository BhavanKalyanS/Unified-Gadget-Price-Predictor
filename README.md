# Unified Gadget Price Predictor (2026 Edition)

A fullstack machine learning web application built on Flask, Scikit-learn, and Tailwind CSS. The app dynamically predicts futuristic 2026 electronics pricing based on hardware capabilities using advanced Regression models (Random Forest, Gradient Boosting, etc).

## Features
- Modular Scikit-learn Machine Learning pipeline
- Synthetic DataFrame Generation functionality
- Flask REST API (`/predict`, `/metrics`)
- Single-page Frontend Architecture with Chart.js

## Project Overview

├── `app.py`                - Core Flask API logic
├── `requirements.txt`      - Dependencies
├── `README.md`
├── `models/`               - Pre-trained ML artifacts (.pkl) and evaluation scores (.json)
├── `datasets/`             - Synthetic datasets used for testing models
├── `templates/`            - Flask HTML views (`index.html`)
├── `static/`               - Frontend dependencies (CSS, JS)
├── `utils/`                - Training logic, Preprocessing logic, Synthetic dataset generator
└── `notebooks/`            - Jupyter Notebook experiments

## Setup
1. Clone / download this directory.
2. Ensure you have Python 3.8+ installed. 
3. Run `pip install -r requirements.txt`.
4. Run `python utils/train_models.py` to bake and evaluate the `.pkl` models and construct `/datasets`.
5. Run `python app.py` to deploy locally on `http://127.0.0.1:5000/`.
