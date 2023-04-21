#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MonotonicityIndex.py
# @Time      :2023/4/21 20:33
# @Author    :LongFei Shan
from typing import Optional, List, Tuple, Union
import numpy as np


def __moveAverage(feature, mode="simple", length=10):
    """
    移动平均
    方法：simple:简单移动平均法
        weight:加权移动平均法
        index:指数加权平均
    其中：l>0
    """
    meanData = []
    if mode == "simple":
        for i in range(len(feature) - length + 1):
            meanData.append(np.mean(feature[i:i+length]))
    if mode == "weight":
        for i in range(len(feature) - length + 1):
            meanData.append(np.sum(np.array(feature[i:i+length])*np.array(range(1, length+1)))/np.sum(np.array(range(1, length+1))))
    if mode == "index":
        index = []
        for j in range(length):
            index.append(0.1*(0.9**j))
        for i in range(len(feature) - length + 1):
            meanData.append(np.sum(np.array(feature[i:i+length])*index)/np.sum(index))
    return meanData


def __deltaFunction(value):
    if value > 0:
        return 1
    else:
        return 0


# 单调性指标
def monotonicityIndex(feature: Union[List[float], np.ndarray, Tuple], mode: str="simple", length: int=10) -> float:
    """
    单调性指标

    :param feature: 特征数据
    :param mode: 移动平均模式
    :param length: 移动平均长度
    :return: 单调性指标
    """
    # 移动平均
    meanData = __moveAverage(feature, mode, length)
    # 计算单调性指标
    delta = []
    for i in range(len(meanData) - 1):
        delta.append(__deltaFunction(meanData[i+1] - meanData[i]))
    return np.sum(delta)