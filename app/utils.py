from PIL import Image
import torch
from .model import preprocess

# Preprocessing images
def prepare_image(file) -> torch.Tensor:
    image = Image.open(file).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    return tensor
