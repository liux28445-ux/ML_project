"""
案例:
    演示 正规方程法线性回归对象 完成 波士顿房价预测案例.
回顾:
    线性回归算法属于有监督学习之 有特征，有标签，且标签是连续的.
    线性回归分类:
        一元线性回归:1个特征列，1个标签列.
        多元线性回归:多个特征列，1个标签列.
    线性回归大白话解释:
        它是用线性公式来描述特征和标签之间关系的，方便做预测，公式如下:
        元线性回归:y=W*x+b
        多元线性回归:y= W1*x1+ W2*x2+ W3*x3+ ...+ Wn*Xn+b= W的转置*x+b
    如何衡量线性回归模型的好坏?
        思路:
            预测值和真实值之间的误差，误差越小，模型越好损失函数
        具体的方案:
        1.最小二乘.         每个(样本)误差平方和
        2.均方误差(MSE)     每个(样本)误差平方和/样本总数
        3.均方根误差(RMSE)  每个(样本)误差平方和/样本总数的平方根
        4.平均绝对误差(MAE)  每个(样本)误差绝对值和/ 样本总数
    如何让损失函数最小?
    思路1:梯度下降法. ==>全梯度下降(Full Gradient Descent，FGD)，随机梯度下降(SGD)，小批量梯度下降(Min-Batch)，随机平均梯度下降(SAG)
    思路2:正规方程法.

    机器学习开发流程:
        1.加载数据.
        2.数据的预处理
        3.特征工程(特征提取，特征预处理...)
        4.模型训练.
        5.模型预测.
        6.模型评估
"""


# from sklearn.datasets import load_boston               #数据
from sklearn.preprocessing import StandardScaler       #特征处理
from sklearn.model_selection import train_test_split   #数据集划分
from sklearn.linear_model import LinearRegression      #正规方程的回归模型
from sklearn.linear_model import SGDRegressor          #梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 均方误差评估
from sklearn.linear_model import Ridge, RidgeCV

# 加载数据
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) # hstack() 函数用于将多个数组（矩阵）水平连接起来，并返回一个新的数组。
target = raw_df.values[1::2, 2]
print(f'特征{data.shape}')
print(f'标签{target.shape}')
#2.数据的预处理
x_train, x_test, y_train, y_test =train_test_split(data, target,test_size=0.2,random_state=23)
#3.特征工程(特征提取，特征预处理...)
# StandardScaler
transder = StandardScaler()
# 标准化
x_train = transder.fit_transform(x_train)
x_test = transder.transform(x_test)
#4.模型训练. 线性回归模型
estimator = LinearRegression(fit_intercept=True)
estimator.fit(x_train, y_train)
print(f'模型的权重{estimator.coef_}')
print(f'模型的偏置{estimator.intercept_}')
# 5.模型预测.
y_per = estimator.predict(x_test)
# 6.模型评估
print(f'均方误差{mean_squared_error(y_test, y_per)}')
print(f'均方根误差{root_mean_squared_error(y_test, y_per)}')
print(f'平均绝对误差{mean_absolute_error(y_test, y_per)}')