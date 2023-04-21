#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :FrequencyDomainFeature.py
# @Time      :2023/4/21 19:49
# @Author    :LongFei Shan
import numpy as np
from typing import Optional, List, Tuple
freqName = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12"] # 频域特征名称


def freDomainFeature(Fre, FFT_y) -> List:
    """频域特征"""
    # 转换Fre与FFT_y为数组
    Fre = np.array(Fre)
    FFT_y = np.array(FFT_y)
    K = len(Fre)
    assert K > 1, "数据个数必须大于1"
    # 频率幅值平均值
    S1 = np.sum(FFT_y) / K
    # 均方根频率
    S2 = np.sqrt(np.sum((FFT_y-S1)**2) / (K-1))
    if S2 == 0:
        S2 = 1e-6
    # S3
    S3 = np.sum(((FFT_y - S1)**3)) / ((K-1) * (S2 ** 3))
    # S4
    S4 = np.sum(((FFT_y - S1)**4)) / ((K-1) * (S2 ** 4))
    # 重心频率
    S5 = np.sum(Fre*FFT_y) / np.sum(FFT_y)
    # S6
    S6 = np.sqrt(np.sum(((Fre - S5)**2)*FFT_y)/(K-1))
    if S6 == 0:
        S6 = 1e-6
    # S7
    S7 = np.sqrt(np.sum((Fre**2)*FFT_y)/np.sum(FFT_y))
    # S8
    S8 = np.sqrt(np.sum((Fre**4)*FFT_y)/np.sum((Fre**2)*FFT_y))
    # S9
    S9 = np.sum((Fre ** 2) * FFT_y) / np.sqrt(np.sum((Fre ** 4) * FFT_y)*np.sum(FFT_y))
    # S10
    S10 = S6 / S5
    # S11
    S11 = np.sum(((Fre - S5)**3)*FFT_y) / ((K-1) * (S6 ** 3))
    # S12
    S12 = np.sum(((Fre - S5)**4)*FFT_y) / ((K-1) * (S6 ** 4))
    return [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12]
