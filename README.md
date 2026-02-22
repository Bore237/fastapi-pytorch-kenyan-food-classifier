# Kenya Food Classification – AI Web Application

A full-stack AI web application that classifies traditional Kenyan food images using a deep learning model deployed with FastAPI.

> 🎯 Goal: Combine Machine Learning, Backend Development, and Web Integration in a production-ready project.

---

## Project Overview

This project is an end-to-end computer vision system capable of classifying Kenyan dishes such as:

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

The model achieves **~80% accuracy** on the validation dataset.

The system includes:

- Deep Learning model (PyTorch)
- REST API built with FastAPI
- Simple web interface for image upload
- 🐳 Deployment-ready backend structure

---

## Architecture

Frontend (HTML/CSS/JS)  
↓  
FastAPI Backend (REST API)  
↓  
PyTorch Model (Food Classifier)

### Workflow

1. User uploads an image
2. Image is sent to FastAPI endpoint
3. Model performs inference
4. Prediction + confidence score returned
5. Result displayed in UI

---

## 🧠 Model Details

- Framework: PyTorch
- Task: Image Classification
- Classes: 13 Kenyan food categories
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Mixed Precision Training (AMP)

### Evaluation Metrics

- Accuracy: ~80%


### Training Features

- Data augmentation
- Learning rate scheduling
- Gradient scaling (AMP)
- Clean modular training pipeline
- Add Grad-CAM visual explanations

---

## ⚙️ Backend – FastAPI

### Features

- `/predict` endpoint
- Image upload handling
- Image preprocessing pipeline
- Model inference
- JSON response with prediction and confidence score

### Example Response

```json
{
  "prediction": "ugali",
  "confidence": 0.87
}
```
---

## Frontend

* Simple responsive UI
* Image preview before submission
* Displays:
    * Predicted class
    * Confidence score
* Built with:
    * HTML
    * CSS
    * JavaScript

---

## Tech Stack

* Python
* PyTorch
* FastAPI
* Uvicorn
* HTML / CSS / JavaScript

---

## Installation

```bash
    git clone https://github.com/Bore237/fastapi-pytorch-kenyan-food-classifier.git
    cd kenya-food-classification-ai

    python -m venv venv
    source venv/bin/activate 

    pip install -r requirements.txt

    uvicorn app.main:app --reload
```
--- 

## Future Improvements

* Docker containerization
* Cloud deployment (AWS / GCP / Azure)
* CI/CD pipeline
* Improve dataset size for higher accuracy

## Author

Goudjou Borel 
Biomedical Engineer and Machine Learning

LinkedIn: [https://www.linkedin.com/in/borel-goudjou-724522262/]