# Author:Administrator
# Name:FFTTransfrom
# Time:2023/4/21  14:14
import numpy as np
import scipy


# 对输入数据以及采样频率进行快速傅里叶变换，返回频率与幅值
def fftTransfrom(data, fs):
    """
    :param data: 输入数据
    :param fs: 采样频率
    :return: 频率，幅值
    """
    # 计算数据长度
    N = len(data)
    # 计算频率
    freq = np.linspace(0, fs/2, int(N / 2))
    # 计算幅值
    amp = 2*np.abs(np.fft.fft(data))[:int(N / 2)] / N
    # 返回频率与幅值
    return freq, amp


def envelopeTransfrom(data, fs):
    '''
    fun: 绘制包络谱图
    param data: 输入数据，1维array
    param fs: 采样频率
    param xlim: 图片横坐标xlim，default = None
    param vline: 图片垂直线，default = None
    '''
    #----去直流分量----#
    data = data - np.mean(data)
    #----做希尔伯特变换----#
    xt = data
    ht = scipy.fftpack.hilbert(xt)
    at = np.sqrt(xt**2+ht**2)   # 获得解析信号at = sqrt(xt^2 + ht^2)
    am = np.fft.fft(at)         # 对解析信号at做fft变换获得幅值
    am = np.abs(am)             # 对幅值求绝对值（此时的绝对值很大）
    am = am/len(am)*2
    am = am[0: int(len(am)/2)]  # 取正频率幅值
    freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(len(freq)/2)]  # 获取正频率
    return freq, am

