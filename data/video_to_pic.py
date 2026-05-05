from pathlib import Path

import cv2
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
vid = PROJECT_ROOT / 'data' / '6517979-hd_1920_1080_24fps.mp4'
output_folder = 'frames'

print(vid)

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(vid)

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f'frame_{frame_id:05d}.jpg')
    cv2.imwrite(filename, frame)

    frame_id += 1

cap.release()
print(f'Saved {frame_id} frames')