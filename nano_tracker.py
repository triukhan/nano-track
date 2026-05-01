import time
from collections import deque
import cv2
import numpy as np
import torch

from utils import corner2center

UPDATE_FREQUENCY = 50
REINIT_SCORE_THRESHOLD = 0.99
SCORE_THRESHOLD = 0.95


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


def normalize(image_patch):
    image_patch = np.ascontiguousarray(
        image_patch.transpose(2, 0, 1)[np.newaxis], dtype=np.float32
    )
    return torch.from_numpy(image_patch)


class NanoTracker:
    def __init__(self, model):
        self.size = None
        self.is_lost = False
        self.center_pos = None
        self.kalman_filter = None
        self.need_init = False
        self.lost_counter = 0
        self.returned_counter = 0
        self.reinit_counter = 0
        self.score_history = deque(maxlen=5)
        self.prev_gray = None
        self.flow_points = None
        self.flow_bbox = None
        self.max_flow_points = 50

        self.context_amount = 0.5
        # --------------------------------------------------------------------------------------------------------------
        self.score_size = 15
        self.stride = 16

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2
        self.window = window.flatten()
        self.points = self.generate_points(self.stride, self.score_size)
        # --------------------------------------------------------------------------------------------------------------
        self._channel_average = None
        self._channel_average_counter = 0
        self.model = model
        self.model.eval()

    def _init_flow_points(self, frame, bbox):
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x1 = max(0, int(x - w / 2))
        y1 = max(0, int(y - h / 2))
        x2 = min(gray.shape[1], int(x + w / 2))
        y2 = min(gray.shape[0], int(y + h / 2))

        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_flow_points,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
            blockSize=7
        )

        self.prev_gray = gray
        self.flow_points = points
        self.flow_bbox = np.array([x, y, w, h], dtype=np.float32)

    def get_subwindow_tracking(self, image, center_pos, model_size, original_size):
        if isinstance(center_pos, float):
            center_pos = [center_pos, center_pos]

        if self._channel_average is None or self._channel_average_counter % 30 == 0:
            self._channel_average = np.mean(image, axis=(0, 1))
        self._channel_average_counter += 1

        im_size = image.shape
        center = (original_size + 1) / 2

        # figure out box original size x original size around center_pos
        context_xmin = np.floor(center_pos[0] - center + 0.5)
        context_xmax = context_xmin + original_size - 1
        context_ymin = np.floor(center_pos[1] - center + 0.5)
        context_ymax = context_ymin + original_size - 1

        # figure out if box beyond the image
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_size[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_size[0] + 1))
        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        # make borders
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = cv2.copyMakeBorder(
                image, top_pad, bottom_pad, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT, value=self._channel_average
            )
            im_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
        else:
            im_patch = image[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]

        # resize patch to size that appropriate for model
        if not np.array_equal(model_size, original_size):
            im_patch = cv2.resize(im_patch, (model_size, model_size))

        im_patch = normalize(im_patch)
        return im_patch

    def init(self, frame, bbox):
        self.lost_counter = 0
        self.kalman_filter = create_kalman()
        self.center_pos = np.array([bbox[0], bbox[1]])
        self.size = np.array([bbox[2], bbox[3]])

        w, h = bbox[2], bbox[3]

        # zeros - we don't have these values so far, cuz this is the first frame
        self.kalman_filter.statePost = np.array([[bbox[0]], [bbox[1]], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kalman_filter.statePre = self.kalman_filter.statePost.copy() # just for correct initializing. by default statePre has zeros
        self.kalman_filter.errorCovPost = np.eye(8, dtype=np.float32)  # set a shape of uncertainty

        # figuring out batch around bbox
        context = self.context_amount * (w + h)
        crop_size = int(np.sqrt((w + context) * (h + context)))  # sqrt saves proportions
        self._init_flow_points(frame, [bbox[0], bbox[1], bbox[2], bbox[3]])

        z_crop = self.get_subwindow_tracking(frame, self.center_pos, 127, crop_size)
        self.model.init(z_crop)
        self.reinit_counter = 0
        self.score_history.clear()

    def track(self, frame):
        self.reinit_counter += 1

        prediction = self.kalman_filter.predict()
        px, py, pw, ph = prediction[:4].flatten()

        flow_result, flow_error = self._track_flow(frame)

        if flow_result is not None:
            fx, fy, fw, fh, flow_n = flow_result
            search_center = np.array([fx, fy], dtype=np.float32)
        else:
            fx, fy, fw, fh, flow_n = None, None, None, None, 0
            search_center = np.array([px, py], dtype=np.float32)

        s_x, scale_z = self._figure_search_size()
        x_crop = self.get_subwindow_tracking(frame, search_center, 255, round(s_x))

        outputs = self.model.track(x_crop)
        scores = self._convert_score(outputs['cls'])
        predicted_bboxes = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(predicted_bboxes[2, :], predicted_bboxes[3, :]) / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        r_c = change((self.size[0] / self.size[1]) / (predicted_bboxes[2, :] / predicted_bboxes[3, :]))

        penalty = np.exp(-(r_c * s_c - 1) * 0.138)
        penalty_scores = penalty * scores
        penalty_scores = penalty_scores * (1 - 0.455) + self.window * 0.455
        best_idx = np.argmax(penalty_scores)

        bbox = predicted_bboxes[:, best_idx] / scale_z

        cx = bbox[0] + search_center[0]
        cy = bbox[1] + search_center[1]

        lr = penalty[best_idx] * scores[best_idx] * 0.348
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, frame.shape[:2])
        score = scores[best_idx]

        dist = np.linalg.norm([cx - px, cy - py])
        norm_dist = dist / (pw + ph + 1e-6)
        motion_penalty = np.exp(-norm_dist * 5)
        conf = penalty[best_idx] * motion_penalty

        tracker_ok = score >= SCORE_THRESHOLD
        self.is_lost = self.lost_counter > 5

        if not tracker_ok:
            self.lost_counter += 1
        elif self.is_lost:
            self.returned_counter += 1 # need to success returning

        print(self.lost_counter, score)

        if not self.is_lost:
            # tracker + flow + kalman
            # weight of tracker depends on score
            wt = np.clip((score - SCORE_THRESHOLD) / (1.0 - SCORE_THRESHOLD + 1e-6), 0.0, 1.0)
            wt = 0.45 + 0.35 * wt
            if flow_result is not None:
                wf = 0.15
                wk = 1.0 - wt - wf
            else:
                wf = 0.0
                wk = 1.0 - wt

            bx = wt * cx + wf * (fx if flow_result is not None else 0.0) + wk * px
            by = wt * cy + wf * (fy if flow_result is not None else 0.0) + wk * py
            bw = wt * width + wf * (fw if flow_result is not None else 0.0) + wk * pw
            bh = wt * height + wf * (fh if flow_result is not None else 0.0) + wk * ph
        else:
            if flow_result is not None and flow_error:
                # tracker is not trusted
                # use only flow + kalman
                print('ONLY FLOW & KALMAN')
                wf = 0.7
                wk = 0.3
                bx = wf * fx + wk * px
                by = wf * fy + wk * py
                bw = wf * fw + wk * pw
                bh = wf * fh + wk * ph
            else:
                print('ONLY KALMAN')
                bx, by, bw, bh = px, py, pw, ph

        bx, by, bw, bh = self._bbox_clip(bx, by, bw, bh, frame.shape[:2])

        if tracker_ok and not self.is_lost or self.returned_counter >= 2:
            self.kalman_filter.correct(
                np.array([[bx], [by], [bw], [bh]], np.float32)
            )
            self.lost_counter = 0
            self.returned_counter = 0
        elif flow_result is not None and flow_n >= 10 and not self.is_lost:
            self.kalman_filter.correct(
                np.array([[bx], [by], [bw], [bh]], np.float32)
            )

        self.score_history.append(score)

        if self.reinit_counter >= UPDATE_FREQUENCY and all(s > REINIT_SCORE_THRESHOLD for s in self.score_history):
            print('RE-INIT')
            self.init(frame, [bx, by, bw, bh])
            return {'filtered': [bx - bw / 2, by - bh / 2, bw, bh]}

        self.center_pos = np.array([bx, by], dtype=np.float32)
        self.size = np.array([bw, bh], dtype=np.float32)

        # reinit flow points periodically
        if (self.reinit_counter % 10 == 0 or self.flow_points is None or len(
                self.flow_points) < 10) and not self.is_lost:
            self._init_flow_points(frame, [bx, by, bw, bh])

        return {'filtered': [bx - bw / 2, by - bh / 2, bw, bh]}

    def _track_flow(self, frame):
        if self.prev_gray is None or self.flow_points is None or len(self.flow_points) < 5:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.flow_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        if next_points is None or status is None:
            self.prev_gray = gray
            self.flow_points = None
            return None, None

        good_old = self.flow_points[status.flatten() == 1].reshape(-1, 2)
        good_new = next_points[status.flatten() == 1].reshape(-1, 2)

        if len(good_new) < 5:
            self.prev_gray = gray
            self.flow_points = None
            return None, None

        displacements = good_new - good_old
        dx, dy = np.median(displacements, axis=0)
        disp_error = np.linalg.norm(displacements - np.median(displacements, axis=0), axis=1)
        median_flow_error = np.median(disp_error)
        print(median_flow_error)

        cx, cy, w, h = self.flow_bbox
        new_cx = cx + dx
        new_cy = cy + dy

        old_center = np.median(good_old, axis=0)
        new_center = np.median(good_new, axis=0)

        old_dist = np.linalg.norm(good_old - old_center, axis=1)
        new_dist = np.linalg.norm(good_new - new_center, axis=1)

        if len(old_dist) > 0 and np.median(old_dist) > 1e-3:
            scale = np.median(new_dist) / np.median(old_dist)
            scale = np.clip(scale, 0.9, 1.1)
        else:
            scale = 1.0

        new_w = w * scale
        new_h = h * scale

        self.prev_gray = gray
        self.flow_points = good_new.reshape(-1, 1, 2)
        self.flow_bbox = np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)

        return (new_cx, new_cy, new_w, new_h, len(good_new)), median_flow_error

    @staticmethod
    def generate_points(stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid(
            [ori + stride * dx for dx in np.arange(0, size)],
            [ori + stride * dy for dy in np.arange(0, size)]
        )
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    @staticmethod
    def _convert_bbox(delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :] #x1
        delta[1, :] = point[:, 1] - delta[1, :] #y1
        delta[2, :] = point[:, 0] + delta[2, :] #x2
        delta[3, :] = point[:, 1] + delta[3, :] #y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    @staticmethod
    def _bbox_clip(cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _figure_search_size(self):
        w_z = self.size[0] + self.context_amount * np.sum(self.size)
        h_z = self.size[1] + self.context_amount * np.sum(self.size)
        area = w_z * h_z
        if area <= 0 or np.isnan(area):
            area = 1.0

        s_z = np.sqrt(area)
        scale_z = 127 / s_z
        s_x = s_z * (255 / 127)
        return s_x, scale_z

    def on_mouse(self, event, cx, cy, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (cx, cy, w, h)
            print(cx, cy)
            self.need_init = True

