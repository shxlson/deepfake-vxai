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

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
    print(f"📸 Found {len(files)} frames")

    for i, fname in enumerate(files):
        print(f"➡️ Processing {i+1}/{len(files)}: {fname}")
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        detect_and_crop(in_path, out_path)

    print("✅ Face detection & cropping completed")
