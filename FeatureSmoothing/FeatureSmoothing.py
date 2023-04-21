# Author:Administrator
# Name:FeatureSmoothing
# Time:2023/4/22  10:50
import numpy as np
import statsmodels.api as sm
from scipy.signal import savgol_filter


def lowessFeatureSmooth(time, Feature, frac=2.0/3.0, it=3, delta=0.0, xvals=None, is_sorted=False, missing='drop', return_sorted=True):
    """
    特征平滑

    :param time: 时间序列
    :param Feature: 特征序列
    :param frac: 0 < frac <= 1, 默认值为2/3
    :param it: 迭代次数，默认值为3
    :param delta: 0 < delta < 1, 默认值为0
    :param xvals: 用于计算的x值，默认值为None
    :param is_sorted: 如果为True，则假定x是已排序的，默认值为False
    :param missing: 如果为'none'，则返回一个异常，如果为'raise'，则引发一个异常，如果为'drop'，则删除缺失值，默认值为'drop'
    :param return_sorted: 如果为True，则返回一个排序的数组，默认值为True
    :return: 平滑后的特征序列
    """
    smoothFeature = sm.nonparametric.lowess(Feature, time, frac=frac, it=it, delta=delta, xvals=xvals, is_sorted=is_sorted, missing=missing, return_sorted=return_sorted)
    return smoothFeature


def savgolFeatureSmooth(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
    """
    特征平滑

    :param x: 特征序列
    :param window_length: 窗口长度
    :param polyorder: 多项式阶数
    :param deriv: 求导的阶数，默认值为0
    :param delta: 采样间隔，默认值为1.0
    :param axis: 沿着哪个轴进行平滑，默认值为-1
    :param mode: 边界模式，默认值为'interp'
    :param cval: 边界填充值，默认值为0.0
    :return: 平滑后的特征序列
    """
    smoothFeature = savgol_filter(x=x, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, axis=axis, mode=mode, cval=cval)
    return smoothFeature


def convoleFeatureSmooth(signal: np.ndarray, kernel: np.ndarray):
    """
    特征平滑

    :param signal: 特征序列
    :param kernel: 卷积核, 一般卷积核为np.ones(len(windowsize)) / len(windowsize), windowsize为卷积核大小，一般为奇数
    :return: 平滑后的特征序列
    """
    smoothFeature = np.convolve(a=signal, v=kernel, mode='full')
    return smoothFeature
