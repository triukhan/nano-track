from pathlib import Path

import cv2
import numpy as np

from kalman import Kalman, H_nano, R_nano, H_lk, R_lk
from models.model_builder import ModelBuilder
from nano_tracker import NanoTracker
from point_tracker import PointTracker
from utils import (load_pretrain)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'nano-track' / 'models' / 'pretrained' / 'nanotrackv3.pth'


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
        self.kalman_filter = Kalman()
        cx, cy, w, h = cbbox

        x = cx - w / 2
        y = cy - h / 2

        bbox = (x, y, w, h)

        # # zeros - we don't have these values so far, cuz this is the first frame
        # self.kalman_filter.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        # self.kalman_filter.statePre = self.kalman_filter.statePost.copy() # just for correct initializing. by default statePre has zeros
        # self.kalman_filter.errorCovPost = np.eye(8, dtype=np.float32)  # set a shape of uncertainty

        self.nano.init(frame, cbbox)
        self.klt.init(frame, bbox)

        s = np.sqrt(w * h)
        self.kalman_filter.x = np.array([cx, cy, 0, 0, s], dtype=np.float32)
        self.bbox = bbox
        self.frame_count = 0
        self.initialized = True
        self.need_init = False

    def track(self, frame: np.ndarray):
        if not self.initialized:
            return
        # if with_kalman:
        self.kalman_filter.predict()

        result = self.nano.track(frame)
        nano_score = result['best_score']
        nano_bbox = result['filtered']

        cx_n, cy_n, w_n, h_n = nano_bbox
        s_n = np.sqrt(w_n * h_n)

        if nano_score > 0.3:
            z_nano = np.array([cx_n, cy_n, s_n], dtype=np.float32)
            self.kalman_filter.update(z_nano, H_nano, R_nano)

        klt_ok, klt_bbox = self.klt.track(frame)
        print(klt_ok)
        if klt_ok:
            cx_k, cy_k, w_k, h_k = klt_bbox
            cx_prev, cy_prev = self.kalman_filter.x[0], self.kalman_filter.x[1]

            vx = cx_k - cx_prev
            vy = cy_k - cy_prev

            z_lk = np.array([vx, vy], dtype=np.float32)

            if np.linalg.norm([vx, vy]) < 100:
                self.kalman_filter.update(z_lk, H_lk, R_lk)

        cx, cy, vx, vy, s = self.kalman_filter.x

        w = s
        h = s
        x = cx - w / 2
        y = cy - h / 2

        self.bbox = (x, y, w, h)

            # nano_ok = nano_score >= self.nano_score_threshold
            # print('nano:', nano_score)
            # if nano_ok:
        #         self.bbox = nano_bbox
        #         self.klt.init(frame, self.bbox)
        #
        #     elif klt_ok:
        #         self.bbox = klt_bbox
        #
        #     else:
        #         return True, self.bbox
        #
        #     # else:
        #     #     return False, None
        #
        # elif klt_ok:

        # return True, self.bbox

    def get_points(self):
        return self.klt.points

    def on_mouse(self, event, cx, cy, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (cx, cy, w, h)
            self.need_init = True
