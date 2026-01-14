# Deepfake VXAI – Video-Level Explainable Deepfake Detection

This repository implements a video-level deepfake detection system with explainability.
The system combines convolutional neural networks, visual attribution (Grad-CAM++),
prototype-based similarity reasoning, and rule-based natural language explanations.

The pipeline is designed to be reproducible, phase-isolated, and suitable for academic submission.

--------------------------------------------------------------------

Repository Structure

deepfake-vxai/
├── data/
│   ├── train/
│   │   ├── raw_videos/
│   │   ├── frames/
│   │   └── faces/
│   ├── test/
│   │   ├── raw_videos/
│   │   ├── frames/
│   │   └── faces/
│   └── inference/
│       ├── raw_videos/
│       ├── frames/
│       └── faces/
│
├── preprocessing/
│   ├── extract_frames.py
│   └── detect_faces.py
│
├── training/
│   ├── train.py
│   └── dataset.py
│
├── explainability/
│   ├── gradcam_pp.py
│   ├── build_prototypes.py
│   └── video_explainer.py
│
├── utils/
├── rules/
├── outputs/
│   ├── checkpoints/
│   └── prototypes/
├── requirements.txt
└── README.md

--------------------------------------------------------------------

Pipeline Overview

Raw Video (.mp4)
      |
      v
Frame Extraction
      |
      v
Face Detection
      |
      v
DenseNet Feature Extraction
      |
      v
Grad-CAM++ Generation
      |
      v
Top-K Frame Selection (per video)
      |
      v
Prototype Similarity (Real vs Fake)
      |
      v
Rule-Based Explanation

--------------------------------------------------------------------

Installation

pip install -r requirements.txt

--------------------------------------------------------------------

Training Phase (Run Once)

1. Place training videos in:
   data/train/raw_videos/
   ├── real/
   └── fake/

2. Extract frames:
   python preprocessing/extract_frames.py --phase train

3. Detect faces:
   python preprocessing/detect_faces.py --phase train

4. Train the model:
   python -m training.train

   Output:
   outputs/checkpoints/densenet.pth

5. Build prototype bank:
   python -m explainability.build_prototypes

   Output:
   outputs/prototypes/real.npy
   outputs/prototypes/fake.npy

--------------------------------------------------------------------

Testing Phase (Optional)

Used only for evaluation on unseen videos.

python preprocessing/extract_frames.py --phase test
python preprocessing/detect_faces.py --phase test

--------------------------------------------------------------------

Inference Phase (Main Usage)

1. Place an unseen video in:
   data/inference/raw_videos/sample_video_01.mp4

2. Extract frames:
   python preprocessing/extract_frames.py --phase inference

3. Detect faces:
   python preprocessing/detect_faces.py --phase inference

4. Run video-level inference and explanation:
   python -m explainability.video_explainer

--------------------------------------------------------------------

Expected Console Output

Running video-level inference & explanation
Loaded trained model weights
Loaded 240 frames from video
Computed Grad-CAM and features
Selected Top-8 frames

Video-Level Decision
Prediction: FAKE
Activation coverage: 0.42
Fake similarity: 0.78
Real similarity: 0.51
Activated frames: 8
Explanation:
Consistent activation observed across multiple frames...

--------------------------------------------------------------------

Notes

- Training, testing, and inference data are strictly isolated.
- Grad-CAM++ is generated only during inference.
- Top-K frame selection is video-specific.
- Debug scripts are excluded from the execution pipeline.

--------------------------------------------------------------------

Summary

This project demonstrates a complete video-level deepfake detection system that not only
classifies videos as real or fake but also explains its decisions using visual evidence
and interpretable reasoning mechanisms.
