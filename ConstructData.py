# Author:Administrator
# Name:ConstructData
# Time:2023/4/23  8:52
# Author:Administrator
# Name:RunProgram
# Time:2023/4/22  13:06
import os
import sys
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from Preprocessing.FFTTransfrom import fftTransfrom
from Preprocessing.Denoising import EMDDenoising
from FeatureExtraction import TimeDomainFeature, FrequencyDomainFeature
from FeatureSmoothing import FeatureSmoothing
from FeatureSelectionIndicators import MonotonicityIndex, SensitivityIndex


def ModuleSegmentation(name):
    # 输出绿色字体 "******************" + f"{name}正在运行..." + "******************"
    tqdm.tqdm.write("\033[1;32m" + "******************" + f"{name}模块---正在运行..." + "******************" + "\033[0m")


def signalDenoisePicture(signal, denoise, signal1, FS, time):
    # 画图,画出去噪前后信号时域与频域图像
    fre1, fft_y1 = fftTransfrom(signal, FS)
    fre2, fft_y2 = fftTransfrom(denoise, FS)
    fre3, fft_y3 = fftTransfrom(signal1, FS)
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 2, 1)
    plt.plot(time, signal)
    plt.title("Original noise signal")
    plt.subplot(3, 2, 2)
    plt.plot(fre1, fft_y1)
    plt.title("Original noise signal frequency domain")
    plt.subplot(3, 2, 3)
    plt.plot(time, denoise)
    plt.title("Denoise signal")
    plt.subplot(3, 2, 4)
    plt.plot(fre2, fft_y2)
    plt.title("Denoise signal frequency domain")
    plt.subplot(3, 2, 5)
    plt.plot(time, signal1)
    plt.title("Original no noise signal1")
    plt.subplot(3, 2, 6)
    plt.plot(fre3, fft_y3)
    plt.title("Original no noise signal1 frequency domain")
    plt.tight_layout()
    plt.show()


def featureExtractionPicture(feature):
    # 画图,画出特征提取后的图像
    # 循环画出合并后feature的特征图像
    featureName = TimeDomainFeature.timeName + FrequencyDomainFeature.freqName
    for i in range(len(feature[0])):
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(feature)[:, i])
        plt.title(f"feature{featureName[i]}")
        plt.tight_layout()
        plt.show()


def featureSelectionPicture(monotonicityIndex, sensitivityIndex, comprehensiveIndex):
    # 画出每个特征的单调性指标与敏感性指标
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(monotonicityIndex)
    plt.title("Monotonicity index")
    plt.subplot(3, 1, 2)
    plt.plot(sensitivityIndex)
    plt.title("Sensitivity index")
    plt.subplot(3, 1, 3)
    plt.plot(comprehensiveIndex)
    plt.title("Comprehensive index")
    plt.tight_layout()
    plt.show()


def saveFeature(feature, name):
    # 保存特征
    tqdm.tqdm.write(f"正在进行特征存储-{name}...")
    feature = pd.DataFrame(feature)
    feature.to_csv(f"./Data/FeatureData/{name}.csv", index=False, sep=",", encoding="utf-8")


def loadTrainData():
    # 读取训练数据
    # dir = r"D:\LongFei-Shan\数据集\bearing\PHM\Test_set"
    dir = r"D:\LongFei-Shan\数据集\bearing\PHM\Learning_set"
    trainDataLifeTime = {"Bearing1_1": 467, "Bearing1_2": 145, "Bearing2_1": 151.67, "Bearing2_2": 132.67, "Bearing3_1": 85.67, "Bearing3_2": 272.67}
    # trainDataLifeTime = {"Bearing1_3": 95.5, "Bearing1_4": 5.65, "Bearing1_5": 26.83, "Bearing1_6": 24.33,
    #                      "Bearing1_7": 126.17, "Bearing2_3": 125.5, "Bearing2_4": 23.17, "Bearing2_5": 51.5,
    #                      "Bearing2_6": 21.5, "Bearing2_7": 9.67, "Bearing3_3": 13.67}
    dirNames = os.listdir(dir)
    trainData = {}
    trainLabel = {}
    # 读取每个文件最后两列数据
    for dirName in dirNames:
        fileNames = os.listdir(f"{dir}/{dirName}")
        data = pd.read_csv(f"{dir}/{dirName}/{fileNames[0]}", sep=",", encoding="utf-8").values
        for fileName in tqdm.tqdm(fileNames[1:], desc=f"读取训练数据进度-{dirName}", ncols=100, file=sys.stdout):
            tempData = pd.read_csv(f"{dir}/{dirName}/{fileName}", sep=",", encoding="utf-8").values
            data = np.vstack((data, tempData))
        trainData[dirName] = data[:, -2:]
        trainLabel[dirName] = np.linspace(0, trainDataLifeTime[dirName], len(data))
    return trainData, trainLabel, dirNames, trainDataLifeTime


def featureExtraction(denoise, windows, FS, dirName, direct):
    feature = []
    for i in tqdm.tqdm(range(len(denoise) // windows + 1), desc=f"特征提取进度-{dirName}-{direct}, ncols=100", file=sys.stdout):
        tempFeature = []
        # 时域特征
        timeDomainFeatures = TimeDomainFeature.timeDomainFeature(denoise[i * windows:(i + 1) * windows])
        # 频域特征
        fre, fft_y = fftTransfrom(denoise[i * windows:(i + 1) * windows], FS)
        frequencyDomainFeatures = FrequencyDomainFeature.freDomainFeature(fre, fft_y)
        # 将时域特征与频域特征合并
        tempFeature.extend(timeDomainFeatures)
        tempFeature.extend(frequencyDomainFeatures)
        feature.append(tempFeature)
    return feature


if __name__ == "__main__":
    # region 读取训练数据
    ModuleSegmentation("读取训练数据")
    trainData, trainLabel, dirNames, trainDataLifeTime = loadTrainData()
    VibrationSignalDirection = ["X", "Y"]
    FS = 25600
    # endregion

    #region 信号去噪
    ModuleSegmentation("信号去噪")
    emd = EMDDenoising("EED", progress=True, waveletName="db6", thresholdName="VisuShrink", thresholdFunctionName="soft", level=6, mode="symmetric", thresholdACF=0.5, exceedThresholdNumber=10)
    denoiseSignal = {}
    windows = 20480
    for dirName in dirNames:
        denoise = {}
        denoise["X"] = []
        denoise["Y"] = []
        for i in tqdm.tqdm(range(len(trainData[dirName]) // windows + 1), desc=f"信号去噪进度-{dirName}", ncols=100, file=sys.stdout):
            # denoise["X"].extend(emd.fit_transfrom(trainData[dirName][i*windows:(i+1)*windows, 0]))
            # denoise["Y"].extend(emd.fit_transfrom(trainData[dirName][i * windows:(i + 1) * windows, 1]))
            denoise["X"].extend(trainData[dirName][i*windows:(i+1)*windows, 0])
            denoise["Y"].extend(trainData[dirName][i*windows:(i+1)*windows, 1])
        denoiseSignal[dirName] = denoise
    # endregion

    #region 特征提取
    ModuleSegmentation("特征提取")
    windows = 1024
    feature = {}
    for dirName in dirNames:
        tempFeatureX = featureExtraction(denoiseSignal[dirName]["X"], windows, FS, dirName, "X")
        tempFeatureY = featureExtraction(denoiseSignal[dirName]["Y"], windows, FS, dirName, "Y")
        feature[dirName] = np.hstack((tempFeatureX, tempFeatureY))
        saveFeature(feature[dirName], f"feature-{dirName}")
    # 将trainLabel分别存储到csv中
    for dirName in tqdm.tqdm(dirNames, desc=f"label-存储进度", ncols=100, file=sys.stdout):
        pd.DataFrame(np.linspace(0, trainDataLifeTime[dirName], len(feature[dirName]))).to_csv(f"./Data/FeatureData/label-{dirName}.csv", index=False, sep=",", encoding="utf-8")
    # endregion

    #region 特征光滑
    ModuleSegmentation("特征光滑")
    for dirName in tqdm.tqdm(dirNames, desc="特征光滑进度", ncols=100, file=sys.stdout):
        feature = pd.read_csv(f"./Data/FeatureData/feature-{dirName}.csv", sep=",", encoding="utf-8").values
        featureSmooth = []
        for subFeature in feature.T:
            tempSmooth = FeatureSmoothing.lowessFeatureSmooth(np.linspace(0, 1, len(subFeature)), subFeature, frac=0.1, it=3)
            featureSmooth.append(tempSmooth[:, 1])
        saveFeature(np.array(featureSmooth).T, f"featureSmooth-{dirName}")
    # endregion

    # #region 特征选择
    # ModuleSegmentation("特征选择")
    # # 计算每个特征的单调性指标与敏感性指标与综合指标
    # monotonicityIndex = {}
    # sensitivityIndex = {}
    # comprehensiveIndex = {}
    # weight = 0.5
    # for dirName in tqdm.tqdm(dirNames, desc="特征选择进度", ncols=100, file=sys.stdout):
    #     feature = pd.read_csv(f"./Data/FeatureData/featureSmooth-{dirName}.csv", sep=",", encoding="utf-8").values
    #     # 计算每个特征的单调性指标与敏感性指标与综合指标
    #     monotonicityIndex[dirName] = []
    #     sensitivityIndex[dirName] = []
    #     comprehensiveIndex[dirName] = []
    #     for i in range(len(feature[0])):
    #         tempMonotonicityIndex = MonotonicityIndex.monotonicityIndex(feature[:, i])
    #         tempSensitivityIndex = SensitivityIndex.sensitivityIndex(feature[:, i])
    #         tempComprehensiveIndex = weight*tempMonotonicityIndex + (1-weight)*tempSensitivityIndex
    #         monotonicityIndex[dirName].append(tempMonotonicityIndex/len(feature[:, i]))  # 归一化，一般单调性指标越大越好
    #         sensitivityIndex[dirName].append(1-tempSensitivityIndex/len(feature[:, i]))  # 归一化，取反，一般敏感性指标越小越好，为了与单调性指标统一因此取反
    #         comprehensiveIndex[dirName].append(tempComprehensiveIndex)
    # # 选择每个dirName下综合指标最大的前10个特征，并取并集
    # featureIndex = []
    # for dirName in dirNames:
    #     tempIndex = np.argsort(comprehensiveIndex[dirName])[-3:]
    #     featureIndex.append(tempIndex)
    # featureIndex = np.array(featureIndex).ravel()
    # featureIndex = np.unique(featureIndex)
    # tqdm.tqdm.write(f"选择的特征索引为：{featureIndex}")
    # # 存储特征索引csv
    # pd.DataFrame(featureIndex).to_csv("./Data/FeatureData/featureIndex.csv", sep=",", encoding="utf-8", index=False, header=False)
    # # endregion

    # region 构造训练数据集
    ModuleSegmentation("构造训练数据集")
    # 读取特征索引
    featureIndex = pd.read_csv("./Data/FeatureData/featureIndex.csv", sep=",", encoding="utf-8", header=None).values.ravel()
    # 读取文件并将smoothData券后读取出来并且合并
    allFeature = pd.read_csv(f"./Data/FeatureData/featureSmooth-{dirNames[0]}.csv", sep=",", encoding="utf-8", header=None).values[1:, featureIndex]
    allLabel = pd.read_csv(f"./Data/FeatureData/label-{dirNames[0]}.csv", sep=",", encoding="utf-8", header=None).values[1:, :]
    for dirName in dirNames[1:]:
        tempfeature = pd.read_csv(f"./Data/FeatureData/featureSmooth-{dirName}.csv", sep=",", encoding="utf-8", header=None).values[1:, featureIndex]
        templabel = pd.read_csv(f"./Data/FeatureData/label-{dirName}.csv", sep=",", encoding="utf-8", header=None).values[1:, :]
        allFeature = np.vstack((allFeature, tempfeature))
        allLabel = np.append(allLabel, templabel)
    # 保存特征与标签
    # pd.DataFrame(allFeature).to_csv("./Data/TestData/TestData.csv", sep=",", encoding="utf-8", index=False, header=False)
    # pd.DataFrame(allLabel).to_csv("./Data/TestData/TestLabel.csv", sep=",", encoding="utf-8", index=False, header=False)
    pd.DataFrame(allFeature).to_csv("./Data/TrainData/trainData.csv", sep=",", encoding="utf-8", index=False, header=False)
    pd.DataFrame(allLabel).to_csv("./Data/TrainData/trainLabel.csv", sep=",", encoding="utf-8", index=False, header=False)
    # endregion

