# Author:Administrator
# Name:RunProgram
# Time:2023/4/22  13:06
import os
import sys
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from LifePredictionModel import LSTMForecast
from ConstructData import ModuleSegmentation


# region 读取训练数据
def readTrainData():
    train_X = pd.read_csv(r".\Data\TrainData\trainData.csv", encoding='utf-8', header=None, sep=',').values
    train_Y = pd.read_csv(r".\Data\TrainData\trainLabel.csv", encoding='utf-8', header=None, sep=',').values
    return train_X, train_Y
# endregion


# region 训练模型
def trainModel(train_X, train_Y):
    model = LSTMForecast.LSTMForeast( lr=0.001, outputDim=1, batchSize=50, epoch=100, timeSteps=1, featureDim=train_X.shape[1], isNormalization=True, featureRange=(0, 1))
    model.fit(train_X, train_Y)


if __name__ == "__main__":
    # region 读取训练数据
    ModuleSegmentation("读取训练数据")
    train_X, train_Y = readTrainData()
    # endregion

    # region 训练模型
    ModuleSegmentation("训练模型")
    trainModel(train_X, train_Y)
    # endregion

    # 加载测试数据
    ModuleSegmentation("加载测试数据")
    test_X = pd.read_csv(r".\Data\FeatureData\featureSmooth-Bearing1_3.csv", encoding='utf-8', sep=',').values
    test_Y = pd.read_csv(r".\Data\FeatureData\label-Bearing1_3.csv", encoding='utf-8', header=None, sep=',').values
    # 加载索引
    index = pd.read_csv(r".\Data\FeatureData\featureIndex.csv", encoding='utf-8', sep=',', header=None).values.ravel()
    # 加载模型
    model = LSTMForecast.LSTMForeast( lr=0.01, outputDim=1, batchSize=50, epoch=100, timeSteps=1, featureDim=train_X.shape[1], isNormalization=True, featureRange=(0, 1))
    result = model.predict(test_X[:, index])

    # 绘制预测结果
    plt.figure()
    plt.plot(result, label="预测结果")
    plt.plot(test_Y, label="真实结果")
    plt.legend()
    plt.show()



