import os
import cv2
from retinaface import RetinaFace
import argparse

INPUT_DIR = "data/frames/sample"
OUTPUT_DIR = "data/faces/sample"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_and_crop(img_path, save_path):
    img = cv2.imread(img_path)
    faces = RetinaFace.detect_faces(img_path)

    if faces:
        face = list(faces.values())[0]
        x1, y1, x2, y2 = map(int, face["facial_area"])
        crop = img[y1:y2, x1:x2]
    else:
        crop = img  # discourse-aware fallback

    crop = cv2.resize(crop, (224, 224))
    cv2.imwrite(save_path, crop)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["train", "test", "inference"],
        help="Pipeline phase to run face detection for"
    )
    args = parser.parse_args()

    PHASE = args.phase

    print(f"Starting face detection for phase: {PHASE}")

    BASE_INPUT = f"data/{PHASE}/frames"
    BASE_OUTPUT = f"data/{PHASE}/faces"

    # In train/test we have labels, in inference we don't
    if PHASE in ["train", "test"]:
        labels = ["real", "fake"]
    else:
        labels = [None]

    for label in labels:
        if label is None:
            in_root = BASE_INPUT
            out_root = BASE_OUTPUT
        else:
            in_root = os.path.join(BASE_INPUT, label)
            out_root = os.path.join(BASE_OUTPUT, label)

        if not os.path.exists(in_root):
            continue

        os.makedirs(out_root, exist_ok=True)

        videos = os.listdir(in_root)
        print(f"Processing {len(videos)} videos")

        for v_idx, video in enumerate(videos):
            in_dir = os.path.join(in_root, video)
            out_dir = os.path.join(out_root, video)
            os.makedirs(out_dir, exist_ok=True)

            frames = [f for f in os.listdir(in_dir) if f.endswith(".jpg")]
            print(f"Video {v_idx+1}/{len(videos)}: {video} ({len(frames)} frames)")

            for f_idx, fname in enumerate(frames):
                print(f"   Frame {f_idx+1}/{len(frames)}", end="\r")

                detect_and_crop(
                    os.path.join(in_dir, fname),
                    os.path.join(out_dir, fname)
                )

            print()  # newline after each video

    print("Face detection completed")



