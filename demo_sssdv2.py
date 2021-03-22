from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import cv2
import torch.utils.data as data
from layers.functions import Detect, PriorBox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from utils.timer import Timer
import xml.etree.ElementTree as ET
import os
import scipy.stats


parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='sssd',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-f', '--file', default='/home/cen/PycharmProjects/TDFSSD/testdata', help='file to run demo')
parser.add_argument('-c', '--camera_num', default=0, type=int,
                    help='demo camera number(default is 0)')
parser.add_argument('-m', '--trained_model', default='/home/cen/PycharmProjects/TDFSSD/weights/sssd_512/0318/sssd_VOC_epoches_146.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval', type=str,
                    help='Dir to save results')
parser.add_argument('-th', '--threshold', default=0.5,
                    type=float, help='Detection confidence threshold value')
parser.add_argument('-t', '--type', dest='type', default='image', type=str,
                    help='the type of the demo file, could be "image", "video", "camera"')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=True, type=bool,
                    help='Use cpu nms')
args = parser.parse_args()

# OS check
# import platform
#
# system_os = platform.system()
# if not system_os == 'Windows':
#     print('ERROR:: This code is for windows OS')
#     sys.exit()

# Make result file saving folder
if not os.path.exists(os.path.join(os.getcwd(), args.save_folder)):
    os.mkdir(os.path.join(os.getcwd(), args.save_folder))

# Config hyper params
VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

mobile_300 = {
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 150, 300],
    'min_sizes': [45, 90, 135, 180, 225, 270],
    'max_sizes': [90, 135, 180, 225, 270, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

Short_VOC_512 = {
    'feature_maps': [64, 32],
    'min_dim': 512,
    'steps': [8, 16],
    'min_sizes': [35.84, 76.8],
    'max_sizes': [76.8, 153.6],
    'aspect_ratios': [[2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}

COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

COCO_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

VOC_SSDVGG_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

COCO_SSDVGG_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}

# Define label map
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

VOC_CLASSES = ('__background__',  # always index 0
               'tassel')

if args.dataset == 'VOC':
    cfg = (VOC_300, Short_VOC_512)[args.size == '512']
    labels = VOC_CLASSES
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']
    labels = COCO_CLASSES

# Version checking
if args.version == 'sssd':
    from models.SSSD import build_net
# elif args.version == 'RFB_E_vgg':
#     from models.RFB_Net_E_vgg import build_net
# elif args.version == 'RFB_mobile':
#     from models.RFB_Net_mobile import build_net
#
#     cfg = mobile_300
# elif args.version == 'DRFB_mobile':
#     from models.DRFB_Net_mobile import build_net
#
#     cfg = mobile_300
# elif args.version == 'SSD_vgg':
#     from models.SSD_vgg import build_net
#
#     cfg = (VOC_SSDVGG_300, COCO_SSDVGG_300)[args.dataset == 'COCO']
# elif args.version == 'SSD_mobile':
#     from models.SSD_lite_mobilenet_v1 import build_net
#
#     cfg = mobile_300
else:
    print('ERROR::UNKNOWN VERSION')
    sys.exit()

# color number book: http://www.n2n.pe.kr/lev-1/color.htm
COLORS = [(255, 0, 0), (153, 255, 0), (0, 0, 255), (102, 0, 0)]  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Prior box setting
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize), interpolation=interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)


def nms_py(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


def demo_img(net, detector, transform, img, save_dir,nms_threhold,score_threhold):
    _t = {'inference': Timer(), 'misc': Timer()}
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if args.cuda:
            x = x.cuda()
            scale = scale.cuda()
    _t['inference'].tic()
    out = net(x)  # forward pass
    boxes, scores = detector.forward(out, priors)
    inference_time = _t['inference'].toc()
    boxes = boxes[0]
    scores = scores[0]
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    _t['misc'].tic()
    for j in range(1, num_classes):
        max_ = max(scores[:, j])
        inds = np.where(scores[:, j] > score_threhold)[0]
        if inds is None:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        # keep = nms(c_dets, args.threshold, force_cpu=args.cpu)
        keep = nms_py(c_dets, nms_threhold)
        c_dets = c_dets[keep, :]
        c_bboxes = c_dets[:, :4]
        total_number = c_bboxes.shape[0]
        for bbox in c_bboxes:
            # Create a Rectangle patch
            label = labels[j]
            score = c_dets[0][4]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
            cv2.putText(img, '{label}'.format(label=label), (int(bbox[0]), int(bbox[1])),
                        FONT, 1, COLORS[1], 2)
    nms_time = _t['misc'].toc()
    # status = ' inference time: {:.3f}s \n nms time: {:.3f}s \n FPS: {:d}'.format(inference_time, nms_time, int(1/(inference_time+nms_time)))
    status = 't_inf: {:.3f} s || t_misc: {:.3f} s || total: {:d} \r'.format(inference_time, nms_time,c_bboxes.shape[0])
    cv2.putText(img, status[:-2], (10, 40), FONT, 1.2, (0, 0, 0), 5)
    cv2.putText(img, status[:-2], (10, 40), FONT, 1.2, (255, 255, 255), 2)
    # print(status)
    cv2.imwrite(save_dir, img)
    return total_number
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    # cv2.destoryAllWindows()


def demo_stream(net, detector, transform, video, save_dir):
    _t = {'inference': Timer(), 'misc': Timer(), 'total': Timer()}

    index = -1
    # avgFPS = 0.0
    while (video.isOpened()):
        _t['total'].tic()
        index = index + 1

        flag, img = video.read()
        # if flag == False: # For fasten loop
        #    break
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if args.cuda:
                x = x.cuda()
                scale = scale.cuda()
        _t['inference'].tic()
        out = net(x)  # forward pass
        boxes, scores = detector.forward(out, priors)
        inference_time = _t['inference'].toc()
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        _t['misc'].tic()
        for j in range(1, num_classes):
            max_ = max(scores[:, j])
            inds = np.where(scores[:, j] > args.threshold)[0]
            # inds = np.where(scores[:, j] > 0.6)[0] # For higher accuracy
            if inds is None:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_dets, args.threshold, force_cpu=False)
            keep = nms_py(c_dets, args.threshold)
            c_dets = c_dets[keep, :]
            c_bboxes = c_dets[:, :4]
            for bbox in c_bboxes:
                # Create a Rectangle patch
                label = labels[j - 1]
                score = c_dets[0][4]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[1], 2)
                cv2.putText(img, '{label}: {score:.2f}'.format(label=label, score=score), (int(bbox[0]), int(bbox[1])),
                            FONT, 1, COLORS[1], 2)
        nms_time = _t['misc'].toc()
        total_time = _t['total'].toc()
        status = 'f_cnt: {:d} || t_inf: {:.3f} s || t_misc: {:.3f} s || t_tot: {:.3f} s  \r'.format(index,
                                                                                                    inference_time,
                                                                                                    nms_time,
                                                                                                    total_time)
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, (0, 0, 0), 5)
        cv2.putText(img, status[:-2], (10, 20), FONT, 0.7, (255, 255, 255), 2)

        cv2.imshow('result', img)
        cv2.waitKey(33)

        cv2.imwrite(os.path.join(save_dir, 'frame_{}.jpg'.format(index)), img)
        sys.stdout.write(status)
        sys.stdout.flush()
def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    # print(gt)
    # print(pd)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae

def compute_mse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    count = 0
    for obj in tree.findall('object'):
        obj_struct = {}
        count = count + 1
        # obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        # obj_struct['difficult'] = int(obj.find('difficult').text)
        # bbox = obj.find('bndbox')
        # obj_struct['bbox'] = [
        #     int(bbox.find('xmin').text),
        #     int(bbox.find('ymin').text),
        #     int(bbox.find('xmax').text),
        #     int(bbox.find('ymax').text)
        # ]
        # objects.append(obj_struct)
    return count

def compute_relerr(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        rmae = np.mean(np.abs(diff) / gt) * 100
        rmse = np.sqrt(np.mean(diff**2 / gt**2)) * 100
    else:
        rmae = 0
        rmse = 0
    return rmae, rmse
def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    pd, gt = np.array(pd), np.array(gt)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
    return r_value**2

if __name__ == '__main__':
    # Validity check
    print('Validity check...')
    # Directory setting
    print('Directory setting...')
    # Setting network
    print('Network setting...')
    img_dim = (300, 512)[args.size == '512']
    num_classes = (2, 81)[args.dataset == 'COCO']
    rgb_means = (104, 117, 123)
    print('Loading pretrained model')
    net = build_net(512, num_classes)  # initialize detector
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    print('Finished loading model')

    detector = Detect(num_classes, 0, cfg)
    transform = BaseTransform(net.size, rgb_means, (2, 0, 1))

    # Running demo
    print('Running demo...')
    weather_config = ['cloud','overcast','sun','total']
    nms_config = [0.1,0.3,0.5]
    score_config = [0.1,0.3,0.5]
    for weather in weather_config:
        for nms_threhold in nms_config:
            for score_threhold in score_config:
                test_path = '/home/cen/dataset/1024data/{0}/testdataset/data'.format(weather)
                result_path = '/home/cen/PycharmProjects/TDFSSD/result_512/weather_{0}/nms_{1}/score_{2}'.format(weather,nms_threhold,score_threhold)
                gt_path = '/home/cen/dataset/1024data/{0}/testdataset/xml'.format(weather)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                gt_list = []
                pred_list = []
                for image in os.listdir(test_path):
                    image_name,_ = image.split('.')
                    xml_name = image_name + '.xml'
                    xml_file = os.path.join(gt_path,xml_name)
                    gt_count = parse_rec(xml_file)

                    image_file = os.path.join(test_path,image)
                    result_file = os.path.join(result_path,image)
                    img = cv2.imread(image_file)
                    number = demo_img(net, detector, transform, img, result_file,nms_threhold,score_threhold)
                    gt_list.append(gt_count)
                    pred_list.append(number)
                mae = str(compute_mae(pd=pred_list, gt=gt_list))
                mase = str(compute_mse(pd=pred_list, gt=gt_list))
                rmae_rmse = str(compute_relerr(pd=pred_list, gt=gt_list))
                r2 = str(rsquared(pd=pred_list, gt=gt_list))
                config_ = 'weather_{0} nms_{1} score_{2} mae:{3} mase:{4} rmae_rmse:{5} r2:{6}'.format(weather,nms_threhold,score_threhold,mae,mase,rmae_rmse,r2)
                config_file = 'weather_{0} nms_{1} score_{2}_.txt'.format(weather,nms_threhold,score_threhold)

                print(config_)
                with open(config_file,'a') as f:
                    f.write(config_)
