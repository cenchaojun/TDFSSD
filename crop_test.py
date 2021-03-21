#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/21 下午4:51
# @Author  : cenchaojun
# @File    : crop_test.py
# @Software: PyCharm
import cv2
import os
import time
def crop_image(src, rownum, colnum, overlap_pix):
    image_name = src.split('/')[-1]
    image_name = image_name[:-4]
    img = cv2.imread(src)
    box_list = []
    w, h = img.shape[1],img.shape[0]
    if rownum <= h and colnum <= w:
        s = os.path.split(src)
        fn = s[1].split('.')
        ext = 'jpg'
        num = 0
        rowheight = h // (rownum - overlap_pix) + 1
        colwidth = w // (colnum - overlap_pix) + 1
        for r in range(rowheight - 1):
            for c in range(colwidth - 1):
                Lx = (c * colnum) - overlap_pix * c
                Ly = (r * rownum) - overlap_pix * r
                if (Lx <= 0):
                    Lx = 0
                if (Ly <= 0):
                    Ly = 0
                Rx = Lx + colnum
                Ry = Ly + rownum
                box = (Lx, Ly, Rx, Ry) #(0,0,512,128)
                box_list.append(box)
                croped_image = img[Ly:Ry,Lx:Rx]
                cv2.imwrite('test.jpg',croped_image)
                # cv2.imshow('croped_image',croped_image)
                # cv2.waitKey(5)
                # cv2.destroyWindow()
                # crop_image = img.crop(box)
                # num = num + 1
                # time.sleep(5)
        print(box_list)
if __name__ == '__main__':
    test_path = '/home/cen/PycharmProjects/TDFSSD/test_large/image'
    row = 1024
    col = 1024
    overlap_pix = 0
    for image in os.listdir(test_path):
        image_file = os.path.join(test_path,image)
        crop_image(src=image_file,rownum=row,colnum=col,overlap_pix=overlap_pix)