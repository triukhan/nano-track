from pathlib import Path

import cv2

from hybrid_tracker import HybridTracker
from models.model_builder import ModelBuilder
from nano_tracker import NanoTracker
from utils import load_pretrain


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nano-track' / 'models' / 'pretrained' / 'nanotrackv3.pth'

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

    tracker = HybridTracker(nano_every=5, nano_score_threshold=0.3, klt_min_points=20)

    cv2.namedWindow('hybrid tracking', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('hybrid tracking', tracker.on_mouse)
    # tracker.on_mouse(cv2.EVENT_LBUTTONDOWN, 166, 221, None, None)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if tracker.need_init:
            tracker.init(frame, tracker.bbox)

        if tracker.bbox is not None:
            _, bbox = tracker.track(frame)
            # tracker.bbox = res['bbox']

            draw_box(frame, bbox, (0, 255, 0), 'bbox')

        cv2.imshow('hybrid tracking', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'nano-track' / 'data' / '8177427-uhd_3840_2160_24fps.mp4'
track_object(vid, stop=True)
