from pathlib import Path

import cv2

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

    model = load_pretrain(ModelBuilder(), MODEL_PATH).eval()
    model.eval()

    tracker = NanoTracker(model)
    cv2.namedWindow('tracking with siam', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('tracking with siam', tracker.on_mouse)
    # tracker.on_mouse(cv2.EVENT_LBUTTONDOWN, 166, 221, None, None)
    count = 0

    while True:
        count += 1
        ret, frame = video.read()
        if not ret:
            break
        #

        # if count in range(10, 100):
        #     print('skip')
        #     continue

        if tracker.need_init:
            tracker.init(frame, tracker.bbox)
            tracker.need_init = False

        if tracker.center_pos is not None:
            res = tracker.track(frame)
            tracker.bbox = res['bbox']

            bbox = res['bbox']
            filtered = res['filtered']
            kalman = res['kalman_prediction']
            sx1, sy1, sx2, sy2 = res['print']
            # topk_idx, scale_z, predicted_bboxes, center_pos, scores = res['topk_idx']
            #
            #
            # for rank, idx in enumerate(topk_idx):
            #     bbox = predicted_bboxes[:, idx] / scale_z
            #
            #     cx = bbox[0] + center_pos[0]
            #     cy = bbox[1] + center_pos[1]
            #     w = bbox[2]
            #     h = bbox[3]
            #
            #     x1 = int(cx - w / 2)
            #     y1 = int(cy - h / 2)
            #     x2 = int(cx + w / 2)
            #     y2 = int(cy + h / 2)
            #
            #     color = (0, 255, 0) if rank == 0 else (255, 0, 0)
            #
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #
            #     cv2.putText(
            #         frame,
            #         f"{scores[idx]:.2f}",
            #         (x1, y1 - 5),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         color,
            #         1
            #     )
            #
            cv2.rectangle(
                frame,
                (int(sx1), int(sy1)),
                (int(sx2), int(sy2)),
                (0, 255, 255),
                2
            )

            x, y, w, h = tracker.bbox
            # draw_box(frame, bbox, (0, 255, 0), "bbox")
            draw_box(frame, filtered, (0, 255, 0), "filtered")
            # draw_box(frame, kalman, (0, 0, 255), "kalman")
            # px, py, pw, ph = res['kalman_prediction']
            # cv2.rectangle(
            #     frame,
            #     (px - pw / 2, py - ph / 2), (pw, ph),
            #     (0, 0, 255),
            #     2
            # )
            # cv2.rectangle(
            #     frame,
            #     (fx - fw / 2, fy - fh / 2), (fw, fh),
            #     (255, 0, 0),
            #     2
            # )
            # x, y, w, h = tracker.bbox
            # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

        cv2.imshow('tracking with siam', frame)

        if stop:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


vid = PROJECT_ROOT / 'nano-track' / 'data' / '8177427-uhd_3840_2160_24fps.mp4'
track_object(vid, stop=True)
