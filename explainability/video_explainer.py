import os
import torch
import numpy as np

from models.densenet import DeepfakeNet
from training.dataset import FrameDataset
from explainability.gradcam_pp import GradCAMPlusPlus
from utils.cam_metrics import compute_activation_coverage, count_activated_frames
from utils.topk import select_topk_by_cam
from utils.similarity import compute_prototype_similarity
from rules.rules import generate_explanation

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VIDEO_NAME = "sample_video_01"   # folder inside data/inference/faces/
TOP_K = 8

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    print("Running video-level inference & explanation")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = DeepfakeNet().to(device)

    model.load_state_dict(torch.load("outputs/checkpoints/densenet.pth", map_location=device))

    model.eval()
    print("Loaded trained model weights")
    print("Model device:", next(model.parameters()).device) #comment this out if necessary


    # -----------------------------
    # Load frames for ONE inference video
    # -----------------------------
    dataset = FrameDataset(
        root_dir=f"data/inference/faces/{VIDEO_NAME}",
        label=0  # dummy label
    )

    if len(dataset) == 0:
        raise RuntimeError("No frames found for inference video")

    print(f"Loaded {len(dataset)} frames from video: {VIDEO_NAME}")

    # -----------------------------
    # Grad-CAM++
    # -----------------------------
    campp = GradCAMPlusPlus(
        model=model,
        target_layer=model.backbone.features
    )

    heatmaps = []
    frame_features = []

    # -----------------------------
    # Forward pass + CAM per frame
    # -----------------------------
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, _ = dataset[idx]
            input_tensor = img.unsqueeze(0).to(device)

            cam = campp.generate(input_tensor)
            heatmaps.append(cam)

            feat = model.backbone.features(input_tensor)
            feat = torch.flatten(feat, start_dim=1)
            frame_features.append(feat.cpu().numpy().squeeze())

    heatmaps = np.array(heatmaps)
    frame_features = np.array(frame_features)

    print("Computed Grad-CAM and features")

    # -----------------------------
    # CAM metrics
    # -----------------------------
    activation_coverage = compute_activation_coverage(heatmaps)
    activated_frames = count_activated_frames(heatmaps)

    # -----------------------------
    # Top-K frame selection (SAME VIDEO)
    # -----------------------------
    topk_indices = select_topk_by_cam(heatmaps, k=TOP_K)
    topk_features = frame_features[topk_indices]

    print(f"Selected Top-{TOP_K} frames")

    # -----------------------------
    # Prototype similarity
    # -----------------------------
    real_prototypes = np.load("outputs/prototypes/real.npy")
    fake_prototypes = np.load("outputs/prototypes/fake.npy")

    fake_similarity, real_similarity = compute_prototype_similarity(
        topk_features,
        real_prototypes,
        fake_prototypes
    )

    # -----------------------------
    # Explanation
    # -----------------------------
    explanation = generate_explanation(
        activation_coverage=activation_coverage,
        fake_similarity=fake_similarity,
        real_similarity=real_similarity,
        activated_frames=activated_frames
    )

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\nVideo-Level Decision")
    print("------------------------------")
    print("Video:", VIDEO_NAME)
    print("Prediction:", "FAKE" if fake_similarity > real_similarity else "REAL")
    print("Activation coverage:", round(activation_coverage, 3))
    print("Fake similarity:", round(fake_similarity, 3))
    print("Real similarity:", round(real_similarity, 3))
    print("Activated frames:", activated_frames)
    print("Explanation:")
    print(explanation)
