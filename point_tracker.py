from pathlib import Path

import cv2
import numpy as np

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def detect_features(gray, bbox):
    margin = 0.2
    x, y, w, h = map(int, bbox)

    x2 = int(x + w * margin)
    y2 = int(y + h * margin)
    w2 = int(w * (1 - 2 * margin))
    h2 = int(h * (1 - 2 * margin))

    mask = np.zeros_like(gray)
    mask[y2:y2 + h2, x2:x2 + w2] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask
    )
    return pts


class PointTracker:
    def __init__(self, min_points: int = 20):
        self.prev_gray = None
        self.points = None
        self.bbox = None
        self.min_points = min_points
        self.initialized = False

    def init(self, frame: np.ndarray, bbox: tuple):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.points = detect_features(gray, bbox)
        self.bbox = tuple(map(float, bbox))
        self.prev_gray = gray.copy()
        self.initialized = self.points is not None and len(self.points) > 0

        return self.initialized

    def track(self, frame: np.ndarray) -> tuple[bool, tuple | None]:
        if not self.initialized or self.points is None:
            return False, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points,
            None,
            **lk_params
        )

        if new_pts is None:
            self.initialized = False
            return False, None

        good_new = new_pts[status == 1]
        good_old = self.points[status == 1]

        if len(good_new) < 6:
            self.initialized = False
            return False, None

        M, inliers = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3
        )

        if M is None:
            self.prev_gray = gray.copy()
            return False, None

        inliers = inliers.ravel().astype(bool)
        good_new = good_new[inliers]

        center_new = np.median(good_new, axis=0)
        x, y, w, h = self.bbox
        x = center_new[0] - w / 2
        y = center_new[1] - h / 2

        self.bbox = (x, y, w, h)
        self.points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray.copy()

        if len(self.points) < self.min_points:
            new_features = detect_features(gray, self.bbox)
            if new_features is not None:
                self.points = new_features

        return True, self.bbox

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
vid = PROJECT_ROOT / 'nano-track' / 'data' / '8177427-uhd_3840_2160_24fps.mp4'

cap = cv2.VideoCapture(vid)
tracker = PointTracker(min_points=20)

bbox = None
need_init = False

def on_mouse(event, x, y, *_):
    global bbox, need_init
    if event == cv2.EVENT_LBUTTONDOWN:
        w, h = 60, 60
        bbox = (x - w//2, y - h//2, w, h)
        need_init = True


while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.namedWindow('tracking', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('tracking', on_mouse)
    if need_init and bbox is not None:
        tracker.init(frame, bbox)
        need_init = False

    success, current_bbox = tracker.track(frame)

    if success:
        x, y, w, h = map(int, current_bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pts = tracker.points
        # if pts is not None:
        #     for p in pts:
        #         cv2.circle(frame, tuple(p.ravel().astype(int)), 3, (0, 255, 0), -1)
    cv2.imshow('tracking', frame)
    if True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()