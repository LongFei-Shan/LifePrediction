#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DataDifference.py
# @Time      :2023/3/8 12:34
# @Author    :LongFei Shan
import numpy as np
import warnings


# 差分
class DataDiff:
    def __init__(self, n):
        # 差分次数
        self.n = n
        # 每次差分舍去的首个数据
        self.firstData = []
        # 每次差分后的结果（包含没有差分的数据）
        self.diffResult = []

    def __diffBase(self, data):
        # 记录第一每次差分第一个舍弃的数据
        self.firstData.append(data[0])
        # 差分
        diffTemp = data[1:] - data[:-1]
        # 记录每次差分结果
        self.diffResult.append(diffTemp)

    def __ndiffBase(self, extraData):
        # 将extraData添加到最后一次差分结果后面
        mixData = np.append(self.diffResult[-1], extraData)
        for i in range(self.n):
            tempData = []
            for j in range(len(mixData)):
                if j == 0:
                    tempData.append(self.firstData[self.n-i-1])
                tempData.append(mixData[j] + tempData[j])
            mixData = tempData

        return mixData[-len(extraData):]

    def fit_transfrom(self, data):
        """
        :param data: array_like
        :return: array_like
        """
        # 判断数据是否为一维数据
        data = np.array(data)
        assert data.ndim == 1
        # 记录差分数据
        self.diffResult.append(data)
        # 差分
        for i in range(self.n):
            self.__diffBase(self.diffResult[i])
        # 差分结果为self.diffResult最后一行数据
        result= self.diffResult[-1]

        return result

    def inverse_transfrom(self, extraData=None):
        # 判断extraData是否为None
        if extraData is None:
            return self.diffResult[0]
        else:
            result = self.__ndiffBase(extraData)
            return result


def diff(arr, n):
    """
    :param arr: 输入数据
    :param n: 差分次数
    :return: 首行数字，差分结果
    """
    warnings.warn("diff is deprecated, you can use DataDiff instand", DeprecationWarning)
    first = []
    dif = []
    if n == 0:
        return None, arr

    first.append(arr[0])
    dif.append(arr[1:] - arr[:-1])
    if n > 1:
        for i in range(n-1):
            first.append(dif[i][0])
            dif.append(dif[i][1:] - dif[i][:-1])
    return first, dif[-1]


def ndiff(first, dif, n, newArr="None"):
    warnings.warn("ndiff is deprecated, you can use DataDiff instand", DeprecationWarning)
    if n == 0:
        return dif, newArr

    arr = []
    for i in range(n):
        temp = []
        for j in range(len(dif[-i - 1])):
            if j == 0:
                temp.append(first[-i - 1])
                temp.append(temp[j] + dif[-i - 1][j])
            else:
                temp.append(temp[j] + dif[-i - 1][j])
        arr.append(temp)

        newtemp = []
        if str(newArr) != "None":
            for k in range(len(newArr)):
                if k == 0:
                    newtemp.append(temp[-1] + newArr[k])
                else:
                    newtemp.append(newtemp[k - 1] + newArr[k])
        newArr = newtemp

    return arr[n - 1], newArr