#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :LSTMForecast.py
# @Time      :2023/3/8 12:34
# @Author    :LongFei Shan
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import LSTMBase
import os
from typing import Tuple, List, Optional, Union


class LSTMForeast:
    def __init__(self, lr: float, outputDim: int, batchSize: int, epoch: int, timeSteps: int, featureDim: int, isNormalization: bool=True, featureRange: Tuple=(0, 1)):
        """

        :param lr: 学习率
        :param outputDim: 输出数据维度
        :param batchSize: 每批次训练数据个数
        :param epoch: 训练次数
        :param timeSteps: 一句话有多少单词
        :param featureDim: 每个单词用多少维度特征表示
        :param isNormalization: 是否进行归一化
        :param featureRange: 归一化范围
        """
        self.lr = lr
        self.outputDim = outputDim
        self.batchSize = batchSize
        self.epoch = epoch
        self.timeSteps = timeSteps
        self.featureDim = featureDim
        self.isNormalization = isNormalization
        self.featureRange = featureRange
        # 加载模型
        self.lstm = LSTMBase.LSTMBase(lr=self.lr, outputDim=self.outputDim, batchSize=self.batchSize, epoch=self.epoch, timeSteps=self.timeSteps, featureDim=self.featureDim)

    def preproccessMethodNormalization(self, data: np.ndarray) -> np.ndarray:
        """

        :param data: 数据
        :param range: 归一化范围
        :return:归一化后的预测数据
        """
        mms = MinMaxScaler(feature_range=self.featureRange)
        try:
            normalResult = mms.fit_transform(data)
        except:
            data = np.array(data).reshape((-1, 1))
            normalResult = mms.fit_transform(data)
            normalResult = np.array(normalResult).ravel()
        # 存储归一化模型
        if not os.path.exists("./NormalizationModel"):
            os.mkdir("./NormalizationModel")
        joblib.dump(mms, "./NormalizationModel/MinMaxScalerNormalization.gz")
        return normalResult

    def constructionData(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> (np.ndarray, np.ndarray):
        """
        :param x: 训练数据
        :param y: 训练数据标签
        :return: 构建好的训练数据
        """
        x_train = np.reshape(x, (-1, self.timeSteps, self.featureDim))
        y_train = np.reshape(y, (-1, self.outputDim))

        return x_train, y_train

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        :param x: 训练数据
        :param y: 训练数据标签
        """
        # 数据预处理
        if self.isNormalization:
            x = self.preproccessMethodNormalization(x)
        # 训练模型
        x_train, y_train = self.constructionData(x, y)
        self.lstm.fit(x_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: 预测数据
        :return: 预测结果
        """
        # 数据预处理
        if self.isNormalization:
            normalization = joblib.load("./NormalizationModel/MinMaxScalerNormalization.gz")
            X = normalization.transform(X)
        # 构建测试数据
        X = np.reshape(X, (-1, self.timeSteps, self.featureDim))
        # 预测
        result = self.lstm.model.predict(X)

        return result

