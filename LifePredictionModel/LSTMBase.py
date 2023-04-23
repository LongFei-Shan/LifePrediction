#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :LSTMBase.py
# @Time      :2023/3/8 12:34
# @Author    :LongFei Shan

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU, BatchNormalization
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
from PredictionPicture import LSTMLossPicture
from tensorflow.keras.utils import plot_model


class LSTMBase:
    def __init__(self, lr, outputDim, batchSize, epoch, timeSteps, featureDim):
        """
        :param lr: 学习率
        :param outputDim: 输出数据维度
        :param batchSize: 每批次训练数据个数
        :param epoch: 训练次数
        :param timeSteps: 一句话有多少单词
        :param featureDim: 每个单词用多少维度特征表示
        """
        # 定义参数
        self.lr = lr
        self.outputDim = outputDim
        self.batchSize = batchSize
        self.epoch = epoch
        self.timeSteps = timeSteps
        self.featureDim = featureDim
        # 定义优化器
        self.optimizer = Adam(learning_rate=self.lr)
        # 定义损失函数
        self.loss = MeanSquaredError()
        # 加载模型
        self.model = self.modelBase()
        # 编译模型
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        # 打印模型结构
        plot_model(self.model, to_file='./model.png', show_shapes=True)

    def modelBase(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.timeSteps, self.featureDim), return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(70))
        model.add(Dense(70, activation="selu"))
        model.add(Dense(30, activation="selu"))
        model.add(Dense(10, activation="selu"))
        model.add(Dense(self.outputDim, activation="sigmoid"))

        return model

    def fit(self, x, y):
        """
        :param x: 训练数据
        :param y: 训练数据标签
        :return:
        """
        # 训练模型
        self.model.fit(x, y, batch_size=self.batchSize, epochs=self.epoch, verbose=2)
        # 存储模型
        if not os.path.exists("./PredictionModel"):
            os.mkdir("./PredictionModel")
        self.model.save("./PredictionModel/LSTMLifePredictionModel/")

    def predict(self, X):
        """
        :param X: 所需预测的数据
        :return: 预测结果
        """
        # 加载模型权重
        model = keras.models.load_model("./PredictionModel/LSTMLifePredictionModel/")
        # 预测模型
        result = model.predict(X)

        return result

