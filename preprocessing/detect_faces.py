import os
import cv2
from retinaface import RetinaFace

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
    print("🔄 Starting face detection...")

    BASE_INPUT = "data/frames"
    BASE_OUTPUT = "data/faces"

    for label in ["real", "fake"]:
        in_root = os.path.join(BASE_INPUT, label)
        out_root = os.path.join(BASE_OUTPUT, label)
        os.makedirs(out_root, exist_ok=True)

        videos = os.listdir(in_root)
        print(f"📁 Processing {label} videos: {len(videos)}")

        for v_idx, video in enumerate(videos):
            in_dir = os.path.join(in_root, video)
            out_dir = os.path.join(out_root, video)
            os.makedirs(out_dir, exist_ok=True)

            frames = [f for f in os.listdir(in_dir) if f.endswith(".jpg")]
            print(f"🎬 [{label}] Video {v_idx+1}/{len(videos)}: {video} ({len(frames)} frames)")

            for f_idx, fname in enumerate(frames):
                print(f"   ➡️ Frame {f_idx+1}/{len(frames)}", end="\r")

                detect_and_crop(
                    os.path.join(in_dir, fname),
                    os.path.join(out_dir, fname)
                )

            print()  # newline after video

    print("✅ Face detection completed")


