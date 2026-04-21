import argparse
import os
from pathlib import Path

import cv2
import torch
import numpy as np

import sys

from nano_tracker import NanoTracker
from dataset import BANDataset
from eval_dataset import GOT10kDataset
from utils import load_pretrain
from eval import eval

sys.path.append(os.getcwd())

from tqdm import tqdm

from models.model_builder import ModelBuilder

parser = argparse.ArgumentParser(description='nanotrack')

parser.add_argument('--dataset', default='GOT-10k', type=str, help='datasets')
parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str)
parser.add_argument('--save_path', default='./results', type=str, help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', action='store_true', help='whether visualzie result')
parser.add_argument('--gpu_id', default='not_set', type=str, help='gpu id')
parser.add_argument('--tracker_name', default='GOT10k', type=str, help='tracker name')
parser.add_argument('--tracker_path', '-p', default='./results', type=str, help='tracker result path')
parser.add_argument('--num', '-n', default=4, type=int, help='number of thread to eval')
parser.add_argument('--show_video_level', '-s', dest='show_video_level', action='store_true')

parser.set_defaults(show_video_level=False)

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)

WINDOW_INFLUENCE = 0.455
PENALTY_K = 0.138
LR = 0.348


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'pretrained' / 'nanotrackv3.pth'
DATASET_PATH = PROJECT_ROOT / 'data'


def main():
    params = [0.0, 0.0, 0.0]

    params[0] = LR
    params[1] = PENALTY_K
    params[2] = WINDOW_INFLUENCE

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, MODEL_PATH).eval()

    # build tracker
    tracker = NanoTracker(model)

    # create dataset
    dataset = GOT10kDataset('10k', DATASET_PATH)

    for v_idx, video in tqdm(enumerate(dataset)):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                # gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]  # [topx,topy,w,h]
                gt_bbox_ = gt_bbox
                tracker.init(img, [cx, cy, w, h ])
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        video_path = os.path.join(args.save_path, args.dataset, args.tracker_name, video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        result_path = os.path.join(video_path,
                                   '{}_time.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in track_times:
                f.write("{:.6f}\n".format(x))
    eval(args)


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h



if __name__ == '__main__':
    main()