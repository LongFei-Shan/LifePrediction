import numpy as np
from typing import Optional, List, Tuple
timeName = ["Mean", "Var", "Energy", "RMS", "STD", "Max", "Min", "Cf", "KurtosisFactor", "SkewnessFactor", "Cs", "ImpulseFactor", "MarginFactor"] # 时域特征名称


def timeDomainFeature(signal) -> List:
    """信号特征"""
    signal = np.array(signal)
    # -------时域特征-------
    # 平均值
    Mean = np.mean(signal)
    # 方差
    Var = np.var(signal, ddof=1)
    # 平均幅值
    MeanAmplitude = np.mean(np.abs(signal))
    # 能量
    Energy = np.sum(np.power(signal, 2))
    # 均方根
    RMS = np.sqrt(np.mean(np.power(signal, 2)))
    # 方根幅值
    SquareRootAmplitude = np.power(np.mean(np.sqrt(np.abs(signal))), 2)
    # 标准差
    STD = np.std(signal, ddof=1)
    # 最大值
    Max = np.mean(np.sort(signal)[-10:])
    # 最小值
    Min = np.mean(np.sort(signal)[:10])

    # -------波形特征-------
    # 峰值
    Peak = np.mean(np.sort(signal)[-10:])
    # 峰值系数
    if RMS == 0:
        Cf = 0
    else:
        Cf = np.mean(np.abs(np.sort(signal)[-10:]))/RMS
    # 峭度
    Kurtosis = np.sum((signal - Mean)**4)
    # 峭度因子
    if STD == 0:
        KurtosisFactor = 0
        SkewnessFactor = 0
    else:
        KurtosisFactor = Kurtosis/((STD**4)*(len(signal - 1)))
        # 偏度因子
        SkewnessFactor = np.sum((np.abs(signal) - Mean) ** 3) / ((STD ** 3) * (len(signal - 1)))
    # 波形系数
    if Mean == 0:
        Cs = 0
    else:
        Cs = RMS/Mean
    # 脉冲因子
    if MeanAmplitude == 0:
        ImpulseFactor = 0
    else:
        ImpulseFactor = Cf/MeanAmplitude
    # 裕度因子
    if SquareRootAmplitude == 0:
        MarginFactor = 0
    else:
        MarginFactor = Cf/SquareRootAmplitude

    return [Mean, Var, Energy, RMS, STD, Max, Min, Cf, KurtosisFactor, SkewnessFactor, Cs, ImpulseFactor, MarginFactor]







