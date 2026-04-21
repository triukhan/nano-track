from pathlib import Path

import cv2
import numpy as np

from models.model_builder import ModelBuilder
from nano_tracker import NanoTracker
from point_tracker import PointTracker
from utils import (load_pretrain)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nano-track' / 'models' / 'pretrained' / 'nanotrackv3.pth'


def create_kalman():
    kalman_f = cv2.KalmanFilter(8, 4)

    kalman_f.transitionMatrix = np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]
    ], np.float32)

    kalman_f.measurementMatrix = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0]
    ], np.float32)

    kalman_f.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
    kalman_f.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    return kalman_f


class HybridTracker:
    def __init__(
        self,
        nano_every: int = 5,
        nano_score_threshold: float = 0.3,
        klt_min_points: int = 20,
    ):
        model = load_pretrain(ModelBuilder(), MODEL_PATH).eval()
        model.eval()

        self.nano = NanoTracker(model)
        self.klt = PointTracker(min_points=klt_min_points)

        self.nano_every = nano_every
        self.nano_score_threshold = nano_score_threshold

        self.bbox = None
        self.frame_count = 0
        self.initialized = False
        self.need_init = False
        self.kalman_filter = None

    def init(self, frame: np.ndarray, cbbox: tuple):
        self.kalman_filter = create_kalman()
        cx, cy, w, h = cbbox

        x = cx - w / 2
        y = cy - h / 2

        bbox = (x, y, w, h)

        # zeros - we don't have these values so far, cuz this is the first frame
        self.kalman_filter.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kalman_filter.statePre = self.kalman_filter.statePost.copy() # just for correct initializing. by default statePre has zeros
        self.kalman_filter.errorCovPost = np.eye(8, dtype=np.float32)  # set a shape of uncertainty

        self.nano.init(frame, cbbox)
        self.klt.init(frame, bbox)

        self.bbox = cbbox
        self.frame_count = 0
        self.initialized = True
        self.need_init = False

    def track(self, frame: np.ndarray, with_kalman=True) -> tuple[bool, tuple | None]:
        if not self.initialized:
            return False, None

        if with_kalman:
            prediction = self.kalman_filter.predict()
            px, py, pw, ph = prediction[:4].flatten()

        self.frame_count += 1
        run_nano = (self.frame_count % self.nano_every == 0)
        klt_ok, klt_bbox = self.klt.track(frame)

        print(klt_ok)

        if run_nano:
            result = self.nano.track(frame)
            nano_score = result['best_score']
            nano_bbox = result['filtered']

            nano_ok = nano_score >= self.nano_score_threshold
            print('nano:', nano_score)
            if nano_ok:
                self.bbox = nano_bbox
                self.klt.init(frame, self.bbox)

            elif klt_ok:
                self.bbox = klt_bbox

            else:
                return True, self.bbox

            # else:
            #     return False, None

        elif klt_ok:
            self.bbox = klt_bbox

        return True, self.bbox

    def get_points(self):
        return self.klt.points

    def on_mouse(self, event, cx, cy, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (cx, cy, w, h)
            self.need_init = True
