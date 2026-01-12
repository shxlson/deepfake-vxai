import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        input_tensor: shape [1, 3, 224, 224]
        returns: Grad-CAM++ heatmap (H, W)
        """
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[0, 0]  # binary classification

        score.backward(retain_graph=True)

        grads = self.gradients        # [C, H, W]
        activations = self.activations

        # ---- Grad-CAM++ weights ----
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)

        eps = 1e-8
        alpha = grads_power_2 / (
            2 * grads_power_2 +
            sum_activations * grads_power_3 + eps
        )

        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3))

        cam = torch.sum(weights[:, :, None, None] * activations, dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze()

        cam -= cam.min()
        cam /= cam.max() + eps

        return cam.cpu().numpy()
