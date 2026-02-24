from PIL import Image
import torch
import cv2
import numpy as np
from .model import preprocess

# Preprocessing images
def prepare_image(file) -> torch.Tensor:
    image = Image.open(file).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    return tensor

# Grad CAM
class GradCAM:
    def __init__(self, model, input_tensor, batch_form=False):
        self.model = model
        self.input_tensor = input_tensor
        self.gradients = None
        self.activations = None
        self.cams = None
        self.fwd_handle = None
        self.bwd_handle = None

    # --- hooks
    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    # --- Grad-CAM operator
    def operate_grad_cam(self):
        # Reset stored values
        self.gradients = None
        self.activations = None

        # Target layer
        target_layer = self.model.layer4[-1].conv3

        # Register hooks
        self.fwd_handle = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self.backward_hook)

        # Forward pass
        self.model.eval()
        output = self.model(self.input_tensor)
        pred_class = output.argmax(1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, pred_class].backward()

        # Remove hooks immediately 
        self.fwd_handle.remove()
        self.bwd_handle.remove()

        # Compute Grad-CAM
        activation = self.activations          # [1, C, H, W]
        gradient = self.gradients              # [1, C, H, W]

        weights = gradient.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        cam = (activation * weights).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # Safe normalization
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        self.cams = cam
        return pred_class

    # --- Save image
    def plot_grad_cam(self, save_path="gradcam_result.jpg"):
        img = self.input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        heatmap = cv2.resize(self.cams, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        cv2.imwrite(save_path, superimposed)
        return superimposed