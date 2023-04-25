import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import os
import joblib


class LinearSVRPrediction:
    def __init__(self, featureRange=(0, 1), isNormalization=True, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        self.featureRange = featureRange
        self.isNormalization = isNormalization
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

    def preproccessMethodNormalization(self, data: np.ndarray, name: str) -> np.ndarray:
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
        joblib.dump(mms, f"./NormalizationModel/MinMaxScalerNormalization-{name}-SVR.gz")
        return normalResult

    def fit(self, x, y):
        if self.isNormalization:
            x = self.preproccessMethodNormalization(x, "x")
            y = self.preproccessMethodNormalization(y, "y")
        model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0, tol=self.tol, C=self.C,
                    epsilon=self.epsilon, shrinking=self.shrinking, cache_size=self.cache_size, verbose=self.verbose,
                    max_iter=self.max_iter)
        model.fit(x, y)
        if not os.path.exists("./PredictionModel/SVRModel"):
            os.mkdir("./PredictionModel/SVRModel")
        joblib.dump(model, "./PredictionModel/SVRModel/LinearSVRModel.gz")

    def predict(self, x):
        if self.isNormalization:
            normalization = joblib.load("./NormalizationModel/MinMaxScalerNormalization-x-SVR.gz")
            x = normalization.transform(x)
        model = joblib.load("./PredictionModel/SVRModel/LinearSVRModel.gz")
        result = model.predict(x)
        if self.isNormalization:
            normalization = joblib.load("./NormalizationModel/MinMaxScalerNormalization-y-SVR.gz")
            result = normalization.inverse_transform(np.reshape(result, (-1, 1))).ravel()
        return result

