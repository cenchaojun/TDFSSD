#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/19 下午7:58
# @Author  : cenchaojun
# @File    : cau_mae.py
# @Software: PyCharm
from math import sqrt
import numpy as np
import xml.etree.ElementTree as ET
import os
import scipy.stats

def mae_value(y_true, y_pred):

    n = len(y_true)
    mae = sum(np.abs(y_true-y_pred))/n
    return mae

def mse_value(y_true, y_pred):
    n = len(y_true)
    mse = sum(np.square(y_true-y_pred))/n
    return mse

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
    gt_path = '/home/cen/dataset/1024data/total/testdataset/xml'
    gt_list = []
    pred_path = '/home/cen/PycharmProjects/TDFSSD/count'
    pred_list = []
    sum_gt = 0
    sum_pred = 0
    for xml in os.listdir(gt_path):
        xml_name,_ = xml.split('.')
        xml_file = os.path.join(gt_path,xml)
        gt_count = parse_rec(xml_file)
        sum_gt = sum_gt + gt_count
        pred_name = xml_name + '.txt'
        pred_file = os.path.join(pred_path,pred_name)
        with open(pred_file,'r') as f:
            pred_count = int(f.read())
        sum_pred = sum_pred + pred_count
        gt_list.append(gt_count)
        pred_list.append(pred_count)
    print(len(gt_list))
    print(len(pred_list))
    print(gt_list)
    print(pred_list)
    print(compute_mae(pd=pred_list,gt=gt_list))
    print(compute_mse(pd=pred_list, gt=gt_list))
    print(compute_relerr(pd=pred_list, gt=gt_list))
    print(rsquared(pd=pred_list, gt=gt_list))
    print(sum_pred)
    print(sum_gt)
