#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SensitivityIndex.py
# @Time      :2023/4/21 20:40
# @Author    :LongFei Shan
import numpy as np
from typing import Optional, List, Tuple, Union


def BoxModel(feature: Union[List[float], np.ndarray, Tuple]) -> Tuple[float, float]:
    """
    箱型图

    :param feature: 特征数据
    :return: DownLimit, UpLimit，上下限阈值
    """
    # 箱型图
    temp = np.sort(feature)
    # 下四分位数
    Q1 = temp[int(len(feature)/4)]
    # 上四分位数
    Q3 = temp[-int(len(feature)/4)]
    # 四分位距离
    IQR = Q3 - Q1
    # 上限
    UpLimit = Q3 + 1.5*IQR
    # 下限
    DownLimit = Q1 - 1.5 * IQR

    return DownLimit, UpLimit


def sensitivityIndex(feature: Union[List[float], np.ndarray, Tuple], length: int=50) -> int:
    """
    灵敏度指标

    :param feature: 特征数据
    :param length: 判断超过阈值上下限的点，并且再改点后连续length个数据超过上下限阈值则返回该点索引
    :return: 灵敏度指标
    """
    # 箱型图
    DownLimit, UpLimit = BoxModel(feature)
    # 计算灵敏度指标
    index = -1
    for i in range(len(feature)):
        if feature[i] > UpLimit or feature[i] < DownLimit:
            if i + length < len(feature):
                for j in range(i, i+length):
                    if feature[j] > UpLimit or feature[j] < DownLimit:
                        index = i
                    else:
                        index = -1
                        break
            else:
                for j in range(i, len(feature)):
                    if feature[j] > UpLimit or feature[j] < DownLimit:
                        index = i
                    else:
                        index = -1
                        break
    return index