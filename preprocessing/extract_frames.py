import os
import sys

def extract_frames(video_path, out_dir, fps=5):
    os.makedirs(out_dir, exist_ok=True)
    
    # Try opencv-python first (most reliable)
    try:
        import cv2
        _extract_with_opencv(video_path, out_dir, fps)
        return
    except ImportError:
        pass
    
    # Try ffmpeg-python
    try:
        import ffmpeg
        _extract_with_ffmpeg_python(video_path, out_dir, fps)
        return
    except ImportError:
        pass
    
    # Fall back to subprocess ffmpeg
    try:
        import subprocess
        import platform
        _extract_with_subprocess(video_path, out_dir, fps)
        return
    except Exception as e:
        print(f"Error: Could not extract frames. Tried opencv-python, ffmpeg-python, and subprocess ffmpeg.", file=sys.stderr)
        print(f"Please install one of: opencv-python, ffmpeg-python, or ensure ffmpeg is in PATH.", file=sys.stderr)
        print(f"Last error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def _extract_with_opencv(video_path, out_dir, fps):
    import cv2
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                output_path = os.path.join(out_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
            
            frame_count += 1
    finally:
        cap.release()
    
    print(f"Extracted {saved_count} frames to {out_dir}")

def _extract_with_ffmpeg_python(video_path, out_dir, fps):
    import ffmpeg
    
    output_pattern = os.path.join(out_dir, "frame_%04d.jpg")
    
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=fps)
        .output(output_pattern)
        .overwrite_output()
        .run(quiet=True)
    )

def _extract_with_subprocess(video_path, out_dir, fps):
    import subprocess
    import platform
    
    output_pattern = os.path.join(out_dir, "frame_%04d.jpg")
    is_windows = platform.system() == "Windows"
    
    # Escape paths for Windows if needed
    if is_windows:
        video_path_escaped = f'"{video_path}"'
        output_pattern_escaped = f'"{output_pattern}"'
        cmd = f'ffmpeg -i {video_path_escaped} -vf fps={fps} {output_pattern_escaped}'
    else:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            output_pattern
        ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        shell=is_windows
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg failed: {error_msg}")

import os

if __name__ == "__main__":
    BASE_INPUT = "data/raw_videos"
    BASE_OUTPUT = "data/frames"

    for label in ["real", "fake"]:
        in_dir = os.path.join(BASE_INPUT, label)
        out_dir = os.path.join(BASE_OUTPUT, label)
        os.makedirs(out_dir, exist_ok=True)

        for video in os.listdir(in_dir):
            if not video.endswith(".mp4"):
                continue

            video_path = os.path.join(in_dir, video)
            video_name = os.path.splitext(video)[0]
            save_dir = os.path.join(out_dir, video_name)

            print(f"🎬 Extracting frames from {label}/{video}")
            extract_frames(video_path, save_dir)
