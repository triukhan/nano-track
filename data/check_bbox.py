from pathlib import Path

import cv2
import os


def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
frames_dir = PROJECT_ROOT / 'data' / 'train' / 'video_40'
gt_path = PROJECT_ROOT / 'data' / 'train' / 'video_40' / 'groundtruth.txt'

image_files = [
    f for f in os.listdir(frames_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
]
image_files = sorted(image_files, key=natural_sort_key)

boxes = []
with open(gt_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.replace(',', ' ').replace('\t', ' ').split()
        if len(parts) >= 4:
            x1, y1, x2, y2 = map(float, parts[:4])
            boxes.append((int(x1), int(y1), int(x2), int(y2)))

n = min(len(image_files), len(boxes))
print(f'frames: {len(image_files)}, boxes: {len(boxes)}, show: {n}')
window_name = 'Ground Truth Viewer'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 700)


for i in range(n):
    img_path = os.path.join(frames_dir, image_files[i])
    img = cv2.imread(img_path)
    if img is None:
        continue

    x1, y1, x2, y2 = boxes[i]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        img,
        f"{i+1}/{n}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow(window_name, img)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()