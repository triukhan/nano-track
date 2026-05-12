import time
from pathlib import Path

import cv2
import torch

from models.model_builder import ModelBuilder
from nano_tracker import NanoTracker
from utils import load_pretrain


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nano-track' / 'models' / 'pretrained' / 'nanotrackv3.pth'

def resize_to_720p_if_needed(frame, max_height=720):
    h, w = frame.shape[:2]

    if h <= max_height:
        return frame, 1.0

    scale = max_height / h
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_bbox(bbox, scale):
    x, y, w, h = bbox
    return [x * scale, y * scale, w * scale, h * scale]


def unscale_bbox(bbox, scale):
    x, y, w, h = bbox
    return [x / scale, y / scale, w / scale, h / scale]


def draw_box(frame, box, color, name):
    x, y, w, h = box

    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))

    cv2.rectangle(frame, pt1, pt2, color, 2)
    cv2.putText(
        frame,
        name,
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def track_object(video_path: Path, stop=False):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = load_pretrain(ModelBuilder(), MODEL_PATH)
    model.eval()

    frame_count = 0
    start_time = time.time()

    tracker = NanoTracker(model)
    cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('tracking', tracker.on_mouse)
    count = 0
    # tracker.on_mouse(cv2.EVENT_LBUTTONDOWN, 551, 1580, 5, 5)

    while True:
        count += 1
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        elapsed = time.time() - start_time
        # fps = frame_count / elapsed if elapsed > 0 else 0q
        # print(f'FPS: {fps:.2f}')

        original_frame = frame.copy()
        resized_frame, scale = resize_to_720p_if_needed(frame, max_height=720)

        if tracker.need_init:
            resized_bbox = scale_bbox(tracker.bbox, scale)
            tracker.init(resized_frame, resized_bbox)
            tracker.need_init = False

        if tracker.center_pos is not None:
            res = tracker.track(resized_frame)
            filtered_resized = res['filtered']
            filtered_original = unscale_bbox(filtered_resized, scale)

            x, y, w, h = filtered_original
            draw_box(original_frame, [x, y, w, h], (0, 255, 0), 'fixed')

        cv2.imshow('tracking', original_frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'nano-track' / 'data' / 'road.mp4'
track_object(vid, stop=True)
