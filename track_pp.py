from pathlib import Path

import cv2

from models.model_builder import ModelBuilder
from nano_tracker import NanoTracker
from utils import load_pretrain


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nano-track' / 'models'    / 'pretrained' / 'nanotrackv3.pth'


def track_object(video_path: Path, stop=False):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    model = load_pretrain(ModelBuilder(), MODEL_PATH).eval()
    model.eval()

    tracker = NanoTracker(model)
    cv2.namedWindow('tracking with siam', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('tracking with siam', tracker.on_mouse)
    # tracker.on_mouse(cv2.EVENT_LBUTTONDOWN, 166, 221, None, None)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if tracker.need_init:
            tracker.init(frame, tracker.bbox)
            tracker.need_init = False

        if tracker.center_pos is not None:
            tracker.bbox = tracker.track(frame)['bbox']
            cv2.rectangle(
                frame,
                (int(tracker.center_pos[0] - 60 / 2), int(tracker.center_pos[1] - 60 / 2)),
                (int(tracker.center_pos[0] - 60 / 2) + 60, int(tracker.center_pos[1] - 60 / 2) + 60),
                (0, 255, 0),
                2
            )
            # x, y, w, h = tracker.bbox
            # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

        cv2.imshow('tracking with siam', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'nano-track' / 'data' / 'helicopter.mp4'
track_object(vid, stop=True)
