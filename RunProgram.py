# Author:Administrator
# Name:RunProgram
# Time:2023/4/22  13:06
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from LifePredictionModel import LSTMForecast, LinearSVRPrediction
from ConstructData import ModuleSegmentation
from sklearn.decomposition import KernelPCA


# region 读取训练数据
def readTrainData():
    train_X = pd.read_csv(r".\Data\TrainData\trainData.csv", encoding='utf-8', header=None, sep=',').values
    train_Y = pd.read_csv(r".\Data\TrainData\trainLabel.csv", encoding='utf-8', header=None, sep=',').values
    return train_X, train_Y
# endregion


# region 训练模型
def trainModel(train_X, train_Y=None, index=None, name="LSTM", flag="train"):
    # if flag=="train":
    #     kpca = KernelPCA(n_components=5, kernel="rbf")
    #     train_X = kpca.fit_transform(train_X)
    #     joblib.dump(kpca, "./kpca.gz")
    # else:
    #     train_X = joblib.load("./kpca.gz").transform(train_X)
    model = {"LSTM":LSTMForecast.LSTMForeast( lr=0.001, outputDim=1, batchSize=50, epoch=30, timeSteps=train_X.shape[1], featureDim=1,
                                              isNormalization=True, featureRange=(0, 1)),
             "SVR":LinearSVRPrediction.LinearSVRPrediction(featureRange=(0, 1), isNormalization=True, kernel='rbf', degree=3, gamma='scale',
                                                             coef0=0.0, tol=1e-3, C=0.5, epsilon=0.1, shrinking=True,
                                                             cache_size=200, verbose=True, max_iter=-1)}
    assert name in model.keys(), "模型名称错误"
    if flag == "train":

        model[name].fit(train_X, train_Y)
    else:
        result = model[name].predict(train_X)
        return result


if __name__ == "__main__":
    # region 读取训练数据
    ModuleSegmentation("读取训练数据")
    train_X, train_Y = readTrainData()
    # endregion

    # region 训练模型
    ModuleSegmentation("训练模型")
    # trainModel(train_X, train_Y.ravel(), name="LSTM", flag="train")
    # endregion

    # 加载测试数据
    ModuleSegmentation("加载测试数据")
    test_X = pd.read_csv(r".\Data\FeatureData\featureSmooth-Bearing3_3.csv", encoding='utf-8', sep=',').values
    test_Y = pd.read_csv(r".\Data\FeatureData\label-Bearing3_3.csv", encoding='utf-8', header=None, sep=',').values
    # 加载索引
    index = pd.read_csv(r".\Data\FeatureData\featureIndex.csv", encoding='utf-8', sep=',', header=None).values.ravel()
    # 测试
    result = trainModel(test_X, name="LSTM", flag="test", index=index)

    # 绘制预测结果
    plt.figure()
    plt.plot(result, label="预测结果")
    plt.plot(test_Y, label="真实结果")
    plt.legend()
    plt.tight_layout()
    plt.show()



