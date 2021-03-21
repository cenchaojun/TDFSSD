#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 上午9:59
# @Author  : cenchaojun
# @File    : model_eval.py
# @Software: PyCharm
from __future__ import print_function

import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from data import VOCroot, COCOroot, VOC_short_512,VOC_300, VOC_512, COCO_300, COCO_512, AnnotationTransform, COCODetection, \
    VOCDetection, detection_collate, BaseTransform, preproc
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='sssd',
                    help='Sorry only TDFSSD_vgg is supported currently!')
parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=True, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max', '--max_epoch', default=150,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('-we', '--warm_epoch', default=50,
                    type=int, help='max epoch for retraining')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--date', default='0210')
parser.add_argument('--save_frequency', default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency', default=10)
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
args = parser.parse_args()

save_folder = os.path.join(args.save_folder, args.version + '_' + args.size, args.date)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
test_save_dir = os.path.join(save_folder, 'ss_predict')
if not os.path.exists(test_save_dir):
    os.makedirs(test_save_dir)

log_file_path = save_folder + '/train' + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log'
if args.dataset == 'VOC':
    train_sets = [('2007', 'train')]
    cfg = (VOC_300, VOC_short_512)[args.size == '512']
else:
    train_sets = [('2017', 'train')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'TDFSSD_vgg':
    from models.TDFSSD_vgg import build_net
elif args.version == 'SSD_HarDNet68':
    from models.SSD_HarDNet68 import build_net
elif args.version == 'sssd':
    from models.SSSD import build_net
else:
    print('Unkown version!')
rgb_std = (1, 1, 1)
img_dim = (300, 512)[args.size == '512']
if 'vgg' in args.version:
    rgb_means = (104, 117, 123)
elif 'mobile' in args.version:
    rgb_means = (103.94, 116.78, 123.68)

p = 0.6
num_classes = (2, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
rgb_means = (104, 117, 123)
gamma = 0.1
momentum = 0.9
if args.visdom:
    import visdom

    viz = visdom.Visdom()

net = build_net(img_dim, num_classes)
#net.load_state_dict()
print(net)
if not args.resume_net:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)


    def xavier(param):
        init.xavier_uniform(param)


    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal_ method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    if args.version == 'FSSD_vgg' or args.version == 'FRFBSSD_vgg' or args.version == 'TDFSSD_vgg':
        net.ft_module.apply(weights_init)
        net.pyramid_ext.apply(weights_init)
    if 'RFB' in args.version:
        net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg' or args.version == 'DenseRFB_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)
    if args.version == 'RANet_vgg':
        net.ft_module.apply(weights_init)

else:
    # load resume network
    resume_net_path = '/home/cen/PycharmProjects/TDFSSD/weights/sssd_512/0318/sssd_VOC_epoches_146.pth'
    # resume_net_path = os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' + \
    #                                str(args.resume_epoch) + '.pth')
    print('Loading resume network', resume_net_path)
    state_dict = torch.load(resume_net_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

detector = Detect(num_classes, 0, cfg)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward())
# dataset
print('Loading Dataset...')
if args.dataset == 'VOC':
    testset = VOCDetection(
        VOCroot, [('2007', 'test')], None, AnnotationTransform())
    train_dataset = VOCDetection(VOCroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p), AnnotationTransform())
elif args.dataset == 'COCO':
    testset = COCODetection(
        COCOroot, [('2017', 'val')], None)
    #testset = COCODetection(COCOroot, [('2017', 'test-dev')], None)
    train_dataset = COCODetection(COCOroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p))
else:
    print('Only VOC and COCO are supported now!')
    exit()


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.5):
    net.eval()
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = 2
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0))
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = False
            else:
                cpu = False

            keep = nms(c_dets, 0.3, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    if args.dataset == 'VOC':
        APs, mAP,rec, prec, ap,Dets,TP,FP,precision_value,recall_value = testset.evaluate_detections(all_boxes, save_folder)
        return APs, mAP,rec, prec, ap,Dets,TP,FP,precision_value,recall_value
    else:
        testset.evaluate_detections(all_boxes, save_folder)
if __name__ == '__main__':
    start_time = time.time()
    test_net(test_save_dir, net, detector, args.cuda, testset,
             BaseTransform(512, rgb_means, rgb_std, (2, 0, 1)),
             300, thresh=0.5)
    end_time = time.time()
    print(end_time-start_time)
