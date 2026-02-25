![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.140.0-green)

# Kenya Food Classification – AI Web Application

A full-stack AI web application that classifies traditional Kenyan food images using a deep learning model deployed with FastAPI.

> 🎯 Goal: Build a production-ready system combining Computer Vision, Backend API, and Interactive Web Interface.

---

## Project Overview

This project is an end-to-end system capable of classifying Kenyan dishes such as:

- Chapati  
- Ugali  
- Nyama Choma  
- Mandazi  
- Sukuma Wiki  
- Pilau  
- Mukimo  
- Githeri  
- Matoke  
- Kachumbari  
- Masala Chips  
- Bhaji  
- Kuku Choma  

The model achieves **~80% accuracy** on the validation dataset and supports Grad-CAM visualizations for interpretability.

**System Components:**

- Deep Learning model (PyTorch)  
- REST API (FastAPI)  
- Web frontend (HTML/CSS/JavaScript) with image upload and prediction display  
- Grad-CAM heatmap visualization

---

## Architecture

```Frontend (HTML/CSS/JS)
↓
FastAPI Backend (REST API)
↓
PyTorch Model (Kenyan Food Classifier)
```

---


### Workflow

1. User uploads an image via the web interface  
2. Image is sent to the FastAPI `/predict` endpoint  
3. PyTorch model performs inference  
4. API returns prediction, confidence scores, and Grad-CAM image  
5. Results displayed on the frontend (prediction list + Grad-CAM)

---

## Model Details

- Framework: PyTorch  
- Task: Image Classification  
- Classes: 13 Kenyan food categories  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Scheduler: ReduceLROnPlateau  
- Mixed Precision Training (AMP)  
- Grad-CAM for interpretability

### Evaluation Metrics

- Accuracy: ~80%  

### Training Features

- Data augmentation (flips, rotations, color jitter)  
- Learning rate scheduling  
- Gradient scaling (AMP)  
- Modular training pipeline  
- Grad-CAM visual explanations  

---

## Backend – FastAPI

### Features

- `/predict` endpoint  
- Handles image uploads (`multipart/form-data`)  
- Performs image preprocessing and model inference  
- Returns JSON with top predictions and Grad-CAM image

### Example Response

```json
{
  "prediction": ["ugali", "chapati", "nyama_choma"],
  "class_prob": [0.87, 0.08, 0.03],
  "gradcam": "<base64-encoded PNG>"
}
```

---

## Frontend

* Responsive UI built with HTML/CSS/JavaScript
* Image preview before submission
* Displays:
    * Top predictions with confidence bars
    * Grad-CAM heatmap
    * Upload + Predict buttons with interactive styling
* Supports multiple predictions with progressive bars

---

## Tech Stack

* Python 3.10
* PyTorch 2.0
* FastAPI 0.140
* Uvicorn
* HTML / CSS / JavaScript

---

## Installation

```bash
git clone https://github.com/Bore237/fastapi-pytorch-kenyan-food-classifier.git
cd kenya-food-classification-ai

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt

uvicorn app.main:app --reload

```
Then open index.html in your browser (or serve via a local web server) to use the app.
---

## Learning & Next Steps

This project was a great opportunity to take my ideas from a Jupyter notebook to a fully functional AI web application. I learned:

- How to build a REST API with FastAPI and integrate it with a deep learning model  
- How to create a responsive web interface with HTML/CSS/JavaScript  
- How to generate Grad-CAM visual explanations for model interpretability  

Next steps I plan to explore:

- Deploying the API on Hugging Face Spaces  
- Hosting the web app on GitHub Pages  
- Deepening my understanding of web app architecture and backend development  

Overall, this project was both fun and educational, helping me bridge the gap between research notebooks and production-ready AI applications.

## Author

Goudjou Borel 
Biomedical Engineer and Machine Learning

LinkedIn: [https://www.linkedin.com/in/borel-goudjou-724522262/]