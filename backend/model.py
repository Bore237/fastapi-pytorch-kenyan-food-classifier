import torch
import os
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# Transformation need per the model
preprocess  = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Make somme adjusment for our custum implementation
num_classes = 13
#model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc = nn.Sequential(
    nn.Dropout(p=0.8),  
    nn.Linear(model.fc.in_features, num_classes)
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
checkpoint_path = os.path.join(BASE_DIR, "checkpoint.pth")
checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()