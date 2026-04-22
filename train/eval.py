import os
import argparse
import sys
from pathlib import Path

sys.path.append(os.path.abspath('.'))

from glob import glob

import warnings

warnings.filterwarnings('ignore')

import six


class GOT10k(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
    """

    def __init__(self, root_dir, subset='test', return_meta=False):
        super(GOT10k, self).__init__()
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'

        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False if subset == 'test' else return_meta
        self._check_integrity(root_dir, subset)

        list_file = os.path.join(root_dir, subset, 'list.txt')  # 路径的拼接
        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')  # 按行读取所有的视频文件夹名
        self.seq_dirs = [os.path.join(root_dir, subset, s)  # 获取所有视频文件的绝对路径
                         for s in self.seq_names]
        self.anno_files = [os.path.join(d, 'groundtruth.txt')
                           for d in self.seq_dirs]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset):
        assert subset in ['train', 'val', 'test']
        list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta

def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious



def eval(args):
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob.glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_name + '*'))

    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    DATASET_PATH = PROJECT_ROOT / 'data'

    root_dir = os.path.abspath(DATASET_PATH)
    e = ExperimentGOT10k(root_dir)
    ao, sr, speed = e.report([args.tracker_name])
    ss = 'ao:%.3f --sr:%.3f -speed:%.3f' % (float(ao), float(sr), float(speed))
    print(ss)

# shell command
# python ./bin/eval.py \
# --tracker_path ./hp_search_result \
# --dataset VOT2018 \
# --num 4 \
# --tracker_name  'checkpoint*'




import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image

fig_dict = {}
patch_dict = {}


def compress(dirname, save_file):
    """Compress a folder to a zip file.

    Arguments:
        dirname {string} -- Directory of all files to be compressed.
        save_file {string} -- Path to store the zip file.
    """
    shutil.make_archive(save_file, 'zip', dirname)


def show_frame(image, boxes=None, fig_n=1, pause=0.001,
               linewidth=3, cmap=None, colors=None, legends=None):
    r"""Visualize an image w/o drawing rectangle(s).

    Args:
        image (numpy.ndarray or PIL.Image): Image to show.
        boxes (numpy.array or a list of numpy.ndarray, optional): A 4 dimensional array
            specifying rectangle [left, top, width, height] to draw, or a list of arrays
            representing multiple rectangles. Default is ``None``.
        fig_n (integer, optional): Figure ID. Default is 1.
        pause (float, optional): Time delay for the plot. Default is 0.001 second.
        linewidth (int, optional): Thickness for drawing the rectangle. Default is 3 pixels.
        cmap (string): Color map. Default is None.
        color (tuple): Color of drawed rectanlge. Default is None.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1])

    if not fig_n in fig_dict or \
            fig_dict[fig_n].get_size() != image.size[::-1]:
        fig = plt.figure(fig_n)
        plt.axis('off')
        fig.tight_layout()
        fig_dict[fig_n] = plt.imshow(image, cmap=cmap)
    else:
        fig_dict[fig_n].set_data(image)

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]

        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + \
                     list(mcolors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        if not fig_n in patch_dict:
            patch_dict[fig_n] = []
            for i, box in enumerate(boxes):
                patch_dict[fig_n].append(patches.Rectangle(
                    (box[0], box[1]), box[2], box[3], linewidth=linewidth,
                    edgecolor=colors[i % len(colors)], facecolor='none',
                    alpha=0.7 if len(boxes) > 1 else 1.0))
            for patch in patch_dict[fig_n]:
                fig_dict[fig_n].axes.add_patch(patch)
        else:
            for patch, box in zip(patch_dict[fig_n], boxes):
                patch.set_xy((box[0], box[1]))
                patch.set_width(box[2])
                patch.set_height(box[3])

        if legends is not None:
            fig_dict[fig_n].axes.legend(
                patch_dict[fig_n], legends, loc=1,
                prop={'size': 8}, fancybox=True, framealpha=0.5)

    plt.pause(pause)
    plt.draw()


class ExperimentGOT10k(object):
    r"""Experiment pipeline and evaluation toolkit for GOT-10k dataset.

    Args:
        root_dir (string): Root directory of GOT-10k dataset where
            ``train``, ``val`` and ``test`` folders exist.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, subset='val',
                 result_dir='results', report_dir='reports'):
        super(ExperimentGOT10k, self).__init__()
        assert subset in ['val', 'test']
        self.subset = subset
        self.dataset = GOT10k(root_dir, subset=subset)
        self.result_dir = os.path.join(result_dir, 'GOT-10k')
        self.report_dir = os.path.join(report_dir, 'GOT-10k')
        self.nbins_iou = 101
        self.repetitions = 3

    def run(self, tracker, visualize=False):
        if self.subset == 'test':
            print('\033[93m[WARNING]:\n' \
                  'The groundtruths of GOT-10k\'s test set is withholded.\n' \
                  'You will have to submit your results to\n' \
                  '[http://got-10k.aitestunion.com/]' \
                  '\nto access the performance.\033[0m')
            time.sleep(2)

        print('Running tracker %s on GOT-10k...' % tracker.name)
        self.dataset.return_meta = False

        # loop over the complete dataset
        for s, (img_files, anno) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (
                s + 1, len(self.dataset), seq_name))

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                        tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue

                # tracking loop
                boxes, times = tracker.track(
                    img_files, anno[0, :], visualize=visualize)

                # record results
                self._record(record_file, boxes, times)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        if self.subset == 'test':
            pwd = os.getcwd()

            # generate compressed submission file for each tracker
            for tracker_name in tracker_names:
                # compress all tracking results
                result_dir = os.path.join(self.result_dir, tracker_name)
                os.chdir(result_dir)
                save_file = '../%s' % tracker_name
                compress('.', save_file)
                print('Records saved at', save_file + '.zip')

            # print submission guides
            print('\033[93mLogin and follow instructions on')
            print('http://got-10k.aitestunion.com/submit_instructions')
            print('to upload and evaluate your tracking results\033[0m')

            # switch back to previous working directory
            os.chdir(pwd)

            return None
        elif self.subset == 'val':
            # meta information is useful when evaluation
            self.dataset.return_meta = True

            # assume tracker_names[0] is your tracker
            report_dir = os.path.join(self.report_dir, tracker_names[0])
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            report_file = os.path.join(report_dir, 'performance.json')

            # visible ratios of all sequences
            seq_names = self.dataset.seq_names
            covers = {s: self.dataset[s][2]['cover'][1:] for s in seq_names}

            performance = {}
            for name in tracker_names:
                print('Evaluating', name)
                ious = {}
                times = {}
                performance.update({name: {
                    'overall': {},
                    'seq_wise': {}}})

                for s, (_, anno, meta) in enumerate(self.dataset):
                    seq_name = self.dataset.seq_names[s]
                    record_files = glob.glob(os.path.join(
                        self.result_dir, name, seq_name,
                        '%s_[0-9]*.txt' % seq_name))
                    if len(record_files) == 0:
                        raise Exception('Results for sequence %s not found.' % seq_name)

                    # read results of all repetitions
                    boxes = [np.loadtxt(f, delimiter=',') for f in record_files]
                    assert all([b.shape == anno.shape for b in boxes])

                    # calculate and stack all ious
                    bound = ast.literal_eval(meta['resolution'])
                    seq_ious = [rect_iou(b[1:], anno[1:], bound=bound) for b in boxes]
                    # only consider valid frames where targets are visible
                    seq_ious = [t[covers[seq_name] > 0] for t in seq_ious]
                    seq_ious = np.concatenate(seq_ious)
                    ious[seq_name] = seq_ious

                    # stack all tracking times
                    times[seq_name] = []
                    time_file = os.path.join(
                        self.result_dir, name, seq_name,
                        '%s_time.txt' % seq_name)
                    if os.path.exists(time_file):
                        seq_times = np.loadtxt(time_file, delimiter=',')
                        seq_times = seq_times[~np.isnan(seq_times)]
                        seq_times = seq_times[seq_times > 0]
                        if len(seq_times) > 0:
                            times[seq_name] = seq_times

                    # store sequence-wise performance
                    ao, sr, speed, _ = self._evaluate(seq_ious, seq_times)
                    performance[name]['seq_wise'].update({seq_name: {
                        'ao': ao,
                        'sr': sr,
                        'speed_fps': speed,
                        'length': len(anno) - 1}})

                ious = np.concatenate(list(ious.values()))
                times = np.concatenate(list(times.values()))

                # store overall performance
                ao, sr, speed, succ_curve = self._evaluate(ious, times)
                performance[name].update({'overall': {
                    'ao': ao,
                    'sr': sr,
                    'speed_fps': speed,
                    'succ_curve': succ_curve.tolist()}})

            # save performance
            with open(report_file, 'w') as f:
                json.dump(performance, f, indent=4)
            # plot success curves
            self.plot_curves([report_file], tracker_names)

            return ao, sr, speed

    def show(self, tracker_names, seq_names=None, play_speed=1):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))

        play_speed = int(round(play_speed))
        assert play_speed > 0
        self.dataset.return_meta = False

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))

            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = np.loadtxt(record_file, delimiter=',')

            # loop over the sequence and display results
            img_files, anno = self.dataset[seq_name]
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('  Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        times = times[:, np.newaxis]
        if os.path.exists(time_file):
            exist_times = np.loadtxt(time_file, delimiter=',')
            if exist_times.ndim == 1:
                exist_times = exist_times[:, np.newaxis]
            times = np.concatenate((exist_times, times), axis=1)
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _check_deterministic(self, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        thr_iou = np.linspace(0, 1, 101)
        bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)

        return ao, sr, speed_fps, succ_curve

    def plot_curves(self, report_files, tracker_names):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot.png')
        key = 'overall'

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='center left',
                           bbox_to_anchor=(1, 0.5))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)



if __name__ == '__main__':
    tracker_name = 'nanotrack'

    dataset = 'DTB70'

    parser = argparse.ArgumentParser(description='tracking evaluation')
    parser.add_argument('--tracker_path', '-p', default='./results', type=str,
                        help='tracker result path')
    parser.add_argument('--dataset', '-d', default=dataset, type=str,
                        help='dataset name')
    parser.add_argument('--num', '-n', default=4, type=int,
                        help='number of thread to eval')
    parser.add_argument('--tracker_name', '-t', default=tracker_name,
                        type=str, help='tracker name')
    parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                        action='store_true')
    parser.set_defaults(show_video_level=False)

    args = parser.parse_args()

    eval(args)



