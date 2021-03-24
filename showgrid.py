#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/23 下午3:06
# @Author  : cenchaojun
# @File    : showgrid.py
# @Software: PyCharm
import cv2
import os
import xml.etree.ElementTree as ET

def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_= cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

image_path = '/home/cen/dataset/origin/cloud/traindataset/data/DJI_0089.JPG'
xml_path = '/home/cen/dataset/origin/cloud/traindataset/data/DJI_0089.xml'
small_pxstep = 512
sx = small_pxstep
sy = small_pxstep
tree = ET.parse(xml_path)
root = tree.getroot()
# 读图像一定要在for循环之前读，要不然重新读，就写不进去框了
img = cv2.imread(image_path)
for bndbox in root.findall('./object/bndbox'):
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=3)
while sx < img.shape[1]:
    cv2.line(img, (sx, 0), (sx, img.shape[0]), color=(168, 132, 61), lineType= cv2.LINE_AA, thickness=8)
    sx += small_pxstep

while sy < img.shape[0]:
    cv2.line(img, (0, sy), (img.shape[1], sy), color=(168, 132, 61), lineType= cv2.LINE_AA, thickness=8)
    sy += small_pxstep
large_pxstep = 1024
lx = large_pxstep
ly = large_pxstep
while lx < img.shape[1]:
    cv2.line(img, (lx+13, 0), (lx+13, img.shape[0]), color=(105, 237, 249), lineType= cv2.LINE_AA, thickness=8)
    lx += large_pxstep

while ly < img.shape[0]:
    cv2.line(img, (0, ly+13), (img.shape[1], ly+13), color=(105, 237, 249), lineType= cv2.LINE_AA, thickness=8)
    ly += large_pxstep
cv2.imwrite('/home/cen/PycharmProjects/TDFSSD/gridiamge7.png', img)

