from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import yaml
import torch
from .model  import model
from .utils import prepare_image

app = FastAPI()

# Lancer Api : uvicorn app.main:app --reload

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # Only for development
]

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

classes = config["classes"]


@app.get("/")
def home():
    return  {"message": "API de classification des nourritures traditionelle du Kenya!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_tensor = prepare_image(file.file)

    # Do the prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        class_id = predicted.item()
    
    class_name = classes[class_id]
    
    return {"prediction": class_name}

