import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from augmentation import Augmentation
from point_target import PointTarget
from utils import center2corner, Center

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

DATASET = {
    'NAMES': ['GOT10K'],
    'VIDEOS_PER_EPOCH': 400000,
    'TEMPLATE': {'SHIFT': 4, 'SCALE': 0.05, 'BLUR': 0.0, 'FLIP': 0.0, 'COLOR': 1.0},
    'SEARCH': {'SHIFT': 64, 'SCALE': 0.18, 'BLUR': 0.2, 'FLIP': 0.0, 'COLOR': 1.0},
    'NEG': 0.2,
    'GRAY': 0.0,
    'GOT10K': {'ROOT': '/home/danylo/data/full_data(1)/train', 'FRAME_RANGE': 100, 'NUM_USE': 100000}
}
TRAIN_EPOCH = 10
EXEMPLAR_SIZE = 127
SEARCH_SIZE = 255
OUTPUT_SIZE = 15



class SubDataset(object):
    def __init__(self, name, root, frame_range, num_use, start_idx):
        self.name = name
        self.root = root
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx

        self.labels = {}
        self.videos = []

        videos = sorted(os.listdir(self.root))

        for video in videos:
            video_path = os.path.join(self.root, video)
            if not os.path.isdir(video_path):
                continue

            gt_path = os.path.join(video_path, "groundtruth.txt")
            if not os.path.exists(gt_path):
                continue

            with open(gt_path, "r") as f:
                lines = f.readlines()

            frames = []
            annos = {}

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                x, y, w, h = map(float, line.split(","))

                if w <= 0 or h <= 0:
                    continue

                # convert to x1,y1,x2,y2
                x1, y1 = x, y
                x2, y2 = x + w, y + h

                frame_id = i
                frames.append(frame_id)
                annos[frame_id] = [x1, y1, x2, y2]

            if len(frames) == 0:
                continue

            self.labels[video] = {
                "0": {
                    "frames": frames,
                    "annos": annos
                }
            }

            self.videos.append(video)

        self.num = len(self.videos)
        self.num_use = self.num if self.num_use == -1 else self.num_use

        self.pick = self.shuffle()

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        image_path = os.path.join(self.root, video, f"{frame+1:08d}.jpg")
        bbox = self.labels[video][track]["annos"][frame]
        return image_path, bbox

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = "0"
        track_info = video[track]

        frames = track_info['frames']

        template_idx = np.random.randint(0, len(frames))
        template_frame = frames[template_idx]

        left = max(template_idx - self.frame_range, 0)
        right = min(template_idx + self.frame_range, len(frames) - 1)

        search_idx = np.random.randint(left, right + 1)
        search_frame = frames[search_idx]

        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)

        video_name = self.videos[index]
        video = self.labels[video_name]
        track = "0"
        track_info = video[track]

        frames = track_info['frames']
        frame = np.random.choice(frames)

        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num

    def log(self):
        logger.info(
            "{} start-index {} select [{}/{}]".format(self.name, self.start_idx, self.num_use, self.num))


class BANDataset(Dataset):
    def __init__(self,):
        super(BANDataset, self).__init__()

        # desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
        #     cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        # if desired_size != cfg.TRAIN.OUTPUT_SIZE:
        #     raise Exception('size not match!')

        # create point target
        self.point_target = PointTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in DATASET['NAMES']:
            subdata_cfg = DATASET.get(name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg['ROOT'],
                    subdata_cfg['FRAME_RANGE'],
                    subdata_cfg['NUM_USE'],
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                DATASET['TEMPLATE']['SHIFT'],
                DATASET['TEMPLATE']['SCALE'],
                DATASET['TEMPLATE']['BLUR'],
                DATASET['TEMPLATE']['FLIP'],
                DATASET['TEMPLATE']['COLOR']
            )
        self.search_aug = Augmentation(
                DATASET['SEARCH']['SHIFT'],
                DATASET['SEARCH']['SCALE'],
                DATASET['SEARCH']['BLUR'],
                DATASET['SEARCH']['FLIP'],
                DATASET['SEARCH']['COLOR']
            )
        videos_per_epoch = DATASET['VIDEOS_PER_EPOCH']
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= TRAIN_EPOCH  # TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = DATASET['GRAY'] and DATASET['GRAY'] > np.random.random()
        neg = DATASET['NEG'] and DATASET['NEG'] > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       SEARCH_SIZE,
                                       gray=gray)

        # get labels
        cls, delta = self.point_target(bbox, OUTPUT_SIZE, neg)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'bbox': np.array(bbox)
                }
