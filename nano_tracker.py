from collections import deque
import cv2
import numpy as np
import torch

from utils import corner2center

UPDATE_FREQUENCY = 50
REINIT_SCORE_THRESHOLD = 0.99


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
    image_patch = image_patch.transpose(2, 0, 1)
    image_patch = image_patch[np.newaxis, :, :, :]
    image_patch = image_patch.astype(np.float32)
    image_patch = torch.from_numpy(image_patch)
    return image_patch


def get_subwindow_tracking(image, center_pos, model_size, original_size):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
    """
    if isinstance(center_pos, float):
        center_pos = [center_pos, center_pos]

    channel_average = np.mean(image, axis=(0, 1))  # figuring out average color if crop_size out of image
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
            image, top_pad, bottom_pad, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT, value=channel_average
        )
        im_patch = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]
    else:
        im_patch = image[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1)]

    # resize patch to size that appropriate for model
    if not np.array_equal(model_size, original_size):
        im_patch = cv2.resize(im_patch, (model_size, model_size))

    im_patch = normalize(im_patch)
    return im_patch


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

        self.model = model
        self.model.eval()

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

        z_crop = get_subwindow_tracking(frame, self.center_pos, 127, crop_size)
        self.model.init(z_crop)
        self.reinit_counter = 0
        self.score_history.clear()

    def track(self, frame):
        self.reinit_counter += 1
        prediction = self.kalman_filter.predict()
        px, py, pw, ph = prediction[:4].flatten()

        w_z = self.size[0] + self.context_amount * np.sum(self.size)
        h_z = self.size[1] + self.context_amount * np.sum(self.size)
        area = w_z * h_z
        if area <= 0 or np.isnan(area):
            area = 1.0

        s_z = np.sqrt(area)
        scale_z = 127 / s_z
        s_x = s_z * (255 / 127)

        x_crop = get_subwindow_tracking(frame, self.center_pos, 255, round(s_x))

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

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        lr = penalty[best_idx] * scores[best_idx] * 0.348
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, frame.shape[:2])

        score = scores[best_idx]
        score_threshold = 0.95

        dist = np.linalg.norm([cx - px, cy - py])
        norm_dist = dist / (pw + ph + 1e-6)
        motion_penalty = np.exp(-norm_dist * 5)
        conf = penalty[best_idx] * motion_penalty

        print(self.lost_counter, score)

        if score < score_threshold:
            self.lost_counter += 1
        elif self.lost_counter >= 5:
            # need to success returning
            self.returned_counter += 1

        self.is_lost = self.lost_counter > 5

        if score > score_threshold and not self.is_lost or self.returned_counter >= 2:
            # detection ok
            self.kalman_filter.correct(np.array([[cx], [cy], [width], [height]], np.float32))
            self.lost_counter = 0
            self.returned_counter = 0

        if self.is_lost:
            if self.lost_counter == 6 or self.lost_counter % 15 == 0:
                best_bbox, best_score, best_center = self._redetect(frame, s_x, scale_z)
                print(best_score)
                if best_score > 0.99:
                    self.lost_counter = 0
                    self.returned_counter = 0
                    self.is_lost = False

                    cx, cy = best_center
                    bw, bh = best_bbox[2], best_bbox[3]
                    self.center_pos = np.array([cx, cy])
                    self.size = np.array([bw, bh])
                    self.kalman_filter.correct(np.array([[cx], [cy], [bw], [bh]], np.float32))
                    self.kalman_filter.statePost[4:8] = 0

                    return {'filtered': best_bbox}

            alpha = 0.0
        else:
            alpha = scores[best_idx] * conf
            alpha = np.clip(alpha, 0, 0.7)

        self.score_history.append(float(score))

        bx = alpha * cx + (1 - alpha) * px
        by = alpha * cy + (1 - alpha) * py
        bw = alpha * width + (1 - alpha) * pw
        bh = alpha * height + (1 - alpha) * ph

        if self.reinit_counter >= UPDATE_FREQUENCY and all(s > REINIT_SCORE_THRESHOLD for s in self.score_history):
            print('RE-INIT')
            self.init(frame, [bx, by, bw, bh])

        self.center_pos = np.array([bx, by])
        self.size = np.array([bw, bh])

        return {'filtered': [bx - bw / 2, by - bh / 2, bw, bh]}

    def _redetect(self, frame, s_x, scale_z):
        H, W = frame.shape[:2]

        cx0, cy0 = self.center_pos

        search_radius = s_x * 2

        x_min = max(0, int(cx0 - search_radius))
        x_max = min(W, int(cx0 + search_radius))
        y_min = max(0, int(cy0 - search_radius))
        y_max = min(H, int(cy0 + search_radius))

        step = int(s_x * 1.2)

        best_score = -1
        best_bbox = None
        best_center = None

        for cy in range(y_min, y_max, step):
            for cx in range(x_min, x_max, step):
                x_crop = get_subwindow_tracking(frame,(cx, cy),255, round(s_x))

                outputs = self.model.track(x_crop)
                scores = self._convert_score(outputs['cls'])
                predicted_bboxes = self._convert_bbox(outputs['loc'], self.points)

                idx = np.argmax(scores)
                bbox = predicted_bboxes[:, idx] / scale_z

                pred_cx = bbox[0] + cx
                pred_cy = bbox[1] + cy
                w = bbox[2]
                h = bbox[3]

                score = scores[idx]

                if score > best_score:
                    best_score = score
                    best_bbox = [pred_cx - w / 2, pred_cy - h / 2, w, h]
                    best_center = (pred_cx, pred_cy)

        return best_bbox, best_score, best_center

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

    def on_mouse(self, event, cx, cy, _, __):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = 60, 60
            self.bbox = (cx, cy, w, h)
            print(cx, cy)
            self.need_init = True

