"""
案例:
    演示逻辑回归模型实现 癌症预测.
逻辑回归模型介绍:
    概述:
        属于有监督学习，即:有特征，有标签，且表示是离散的.
    主要适用于:  二分类.
    原理:
        把线性回归处理后的预测值通过 Sigmoid激活函数，映射到[0，1]概率 基于自定义的阀值，结合概率来分类.损失函数:
        极大似然估计函数的负数形式.
    回顾:机器学习项目流程
        1.加载数据
        2.数据预处理.
        3.特征工程(预处理...)
        4.模型训练
        5.模型预测.
        6.模型评估.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression # 逻辑回归模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # 模型评估
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./data/breast-cancer-wisconsin.csv")

# print(data.info())

# 数据预处理
# 将缺失值替换为np.nan
data = data.replace("?", np.nan)
data.dropna(inplace=True)

# 特征工程 (提取...预处理...)
x  = data.iloc[:,1:-1] # 获取特征列,: 所有行.1-1列：从第1列开始，到倒数第2列结束
# y = data.iloc[:,-1] # 获取标签列
y = data.Class # 获取标签列,效果同上

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 创建标准化对象
transfer = StandardScaler()
# 训练
transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 创建逻辑回归对象
estimator = LogisticRegression()

# 训练
estimator.fit(x_train, y_train)
y_pre = estimator.predict(x_test)
print("准确率:", accuracy_score(y_test, y_pre))
print("权重:", estimator.coef_)
print("截距:", estimator.intercept_)