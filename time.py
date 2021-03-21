#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/16 下午4:17
# @Author  : cenchaojun
# @File    : time.py
# @Software: PyCharm
import time

now_time = time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time()))
print(repr(now_time))