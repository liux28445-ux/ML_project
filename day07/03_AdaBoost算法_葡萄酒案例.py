"""
案例:
    演示AdaBo0st算法 之 葡萄酒案例.
AdaBo0st算法介绍:
    它属于Boosting思想，即:串行执行，每次使用全部样本，最后加权投票.
    原理:
        1.使用全部样本，通过决策树模型(第1个弱分类)进行训练，获取结果.
        思路:
            预测正确 ->  权重下降
            预测错误 ->  权重上升
        2.把第1个弱分类器的处理结果，交给第2个弱分类器进行训练，获取结果.
        思路:
            预测正确 -> 权重下降
            预测错误 -> 权重上升
        3.依次类推，串行执行，直至获取最终结果.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder  # 标签编码器
from sklearn.model_selection import train_test_split # 训练集、测试集分割
from sklearn.tree import DecisionTreeClassifier # 决策树分类器
from sklearn.ensemble import AdaBoostClassifier # Adaboost分类器 -> 集成学习Boosting思想
from sklearn.metrics import  accuracy_score # 模型评估 -> 正确率

# 获取数据集
df_wine  = pd.read_csv('./data/红酒品质分类.csv')
df_wine.info ()