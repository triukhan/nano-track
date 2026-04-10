import numpy as np

from siam_tracker.utils import corner2center

STRIDE = 16
SEARCH_SIZE = 255
OUTPUT_SIZE = 15
NEG_NUM = 16
POS_NUM = 16
TOTAL_NUM = 64


class Point:
    """
    This class generate points.
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride
        self.size = size
        self.image_center = image_center

        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)

        return points



class PointTarget:
    def __init__(self,):
        self.points = Point(STRIDE, OUTPUT_SIZE, SEARCH_SIZE//2)

    def __call__(self, target, size, neg=False):

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        if neg:
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
            neg, neg_num = select(neg, NEG_NUM)
            cls[neg] = 0

            return cls, delta

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                       np.square(tcy - points[1]) / np.square(th / 4) < 1)
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / 2) +
                       np.square(tcy - points[1]) / np.square(th / 2) > 1)
        
        # sampling
        pos, pos_num = select(pos, POS_NUM)
        neg, neg_num = select(neg, TOTAL_NUM - POS_NUM)

        cls[pos] = 1
        cls[neg] = 0

        return cls, delta
