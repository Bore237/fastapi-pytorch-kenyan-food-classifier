weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

num_classes = 13
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.fc = nn.Sequential(
    nn.Dropout(p=0.8),  
    nn.Linear(model.fc.in_features, num_classes)
)
checkpoint = torch.load("checkpoint.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])