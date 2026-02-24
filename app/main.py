from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml
from PIL import Image
import torch
import torch.nn.functional as F
from backend.model  import model
from backend.utils import prepare_image, GradCAM
import io
import base64

# Lancer Api : uvicorn app.main:app --reload
app = FastAPI()

# Peut etre utiliser pas tout les navigateur
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

# Read config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
classes = config["classes"]


def generate_gradcam(image_tensor):
    grad_cam = GradCAM(model, image_tensor)
    _ = grad_cam.operate_grad_cam()
    grad_img = grad_cam.plot_grad_cam()

    # GradCAM
    img_grad = Image.fromarray(grad_img)
    buffered_grad = io.BytesIO()
    img_grad.save(buffered_grad, format="PNG") 
    img_grad_base64 = base64.b64encode(buffered_grad.getvalue()).decode()

    return img_grad_base64

@app.get("/")
def home():
    return  {"message": "API de classification des nourritures traditionelle du Kenya!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_tensor = prepare_image(file.file)

        # Do the prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            top5_probs, top5_indices = torch.topk(probs, 5)

        class_probs = top5_probs[0].tolist()
        top_indices = top5_indices[0].tolist()
        class_names = [classes[i] for i in top_indices]

        gradcam_image = generate_gradcam(image_tensor)

        return {
            "prediction": class_names,
            "class_prob": class_probs,
            "gradcam": gradcam_image,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)