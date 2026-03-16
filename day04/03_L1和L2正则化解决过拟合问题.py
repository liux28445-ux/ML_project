"""
案例:
    演示欠拟合，正好拟合，过拟合，L1正则化，L2正则化的 效果图.
回顾:
    欠拟合:模型在训练集 和 测试集表现效果都不好.
    正好拟合:模型在训练集和 测试集表现效果都好.
    过拟合:模型在训练集表现好，测试集表现不好.
过拟合，欠拟合解释:
    产生原因:
        欠拟合:模型简单.
        过拟合:模型复杂。
    解决方案:
        欠拟合:增加特征，从而增加 模型的复杂度。
        过拟合:减少模型复杂度，手动筛选(减少)特征，L1和L2正则化。
L1和L2正则化介绍:
    目的/思路:
        都是基于惩罚系数来修改(特征列的)权重的，惩罚系数越大，则修改力度就越大，对应的权重就越小.
    区别:
        L1正则化，可以实现让权重变为0，从而达到 特征选择的目的.
        L2正则化，，只能让权重无限趋近于0，但是不能为0.
    大白话:
        我要去爬山，带了个小包，装了:登山杖，水，面包，衣服，雨伞，鞋子...发现包装不下了.
        L1正则化:可以实现去掉一些不是必选的，例如:当天去，当前回，且天气晴朗不带雨伞，鞋子，即:权重为0
        L2正则化:换一个非常非常大的包，还是那些物品，但是空间占用(权重)就变小了...
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    #数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error # 均方误差, RmsE, MAE
from sklearn.linear_model import Ridge, RidgeCV,Lasso


# 欠拟合
def dm01_under_fitting():
    # 指定随机种子 ，确保每次生成的数据集都是相同的
    np.random.seed(100)
    x = np.random.uniform(-3, 3, size=100) # 随机生成100个[-3, 3]之间的数 模拟特征
    # 生成标签
    # 标签方程为 y = 0.5 * x + 2 + 噪声
    y = 0.5 * x + 2 + np.random.normal(0, 1, size=100) # 参1：均值，参2：标准差，参3：生成个数

    # 数据预处理，把x轴（特征）转成 多行 1列
    X = x.reshape(-1, 1)

    # 特征工程： polynomial
    estmitor = LinearRegression(fit_intercept= True)
    # 训练
    estmitor.fit(X, y)
    # print('权重：', estmitor.coef_)
    # print('截距：', estmitor.intercept_)

    # 预测
    y_predict = estmitor.predict(X)

    # 评估
    print('均方误差：', mean_squared_error(y, y_predict))
    print('RmsE：', root_mean_squared_error(y, y_predict))
    print('平均绝对误差：', mean_absolute_error(y, y_predict))

    # 可视化
    plt.scatter(x, y) # 绘制散点图
    plt.plot(x, y_predict, color='r')
    plt.show()

# 正好拟合
def dm02_just_fitting():
    # 指定随机种子 ，确保每次生成的数据集都是相同的
    np.random.seed(100)
    x = np.random.uniform(-3, 3, size=100) # 随机生成100个[-3, 3]之间的数 模拟特征
    # 生成标签
    # 标签方程为 y = 0.5 * x **2+x+ 2 + 噪声
    y = 0.5 * x **2+x+ 2 + np.random.normal(0, 1, size=100) # 参1：均值，参2：标准差，参3：生成个数

    # 数据预处理，把x轴（特征）转成 多行 1列
    X = x.reshape(-1, 1)

    #因为目前特征列只有1列，模型过于简单，会出现欠拟合的问题，我们增加1列 特征列，从而增加模型的复杂度.
    # 创建一个特征列
    X2 = np.hstack([X, X**2]) # 该函数作用：创建一个新特征列，进行水平拼接，该列是输入特征列的平方
    # 特征工程： polynomial
    estmitor = LinearRegression()
    # 训练
    estmitor.fit(X2, y)
    # print('权重：', estmitor.coef_)
    # print('截距：', estmitor.intercept_)

    # 预测
    y_predict = estmitor.predict(X2)

    # 评估
    print('均方误差：', mean_squared_error(y, y_predict))
    # print('RmsE：', root_mean_squared_error(y, y_predict))
    # print('平均绝对误差：', mean_absolute_error(y, y_predict))

    # 可视化
    plt.scatter(x, y) # 绘制散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()

# 过拟合
def dm03_over_fitting():
    # 指定随机种子 ，确保每次生成的数据集都是相同的
    np.random.seed(100)
    x = np.random.uniform(-3, 3, size=100)  # 随机生成100个[-3, 3]之间的数 模拟特征
    # 生成标签
    # 标签方程为 y = 0.5 * x **2+x+ 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数

    # 数据预处理，把x轴（特征）转成 多行 1列
    X = x.reshape(-1, 1)

    # 因为目前特征列只有1列，模型过于简单，会出现欠拟合的问题，我们增加1列 特征列，从而增加模型的复杂度.
    # 创建一个特征列
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5,X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 该函数作用：创建一个新特征列，进行水平拼接，该列是输入特征列的平方
    # 特征工程： polynomial
    estmitor = LinearRegression()
    # 训练
    estmitor.fit(X3, y)
    # print('权重：', estmitor.coef_)
    # print('截距：', estmitor.intercept_)

    # 预测
    y_predict = estmitor.predict(X3)

    # 评估
    print('均方误差：', mean_squared_error(y, y_predict))
    # print('RmsE：', root_mean_squared_error(y, y_predict))
    # print('平均绝对误差：', mean_absolute_error(y, y_predict))

    # 可视化
    plt.scatter(x, y)  # 绘制散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()
# L1正则化
def dm04_l1_regularization():
    # 指定随机种子 ，确保每次生成的数据集都是相同的
    np.random.seed(100)
    x = np.random.uniform(-3, 3, size=100)  # 随机生成100个[-3, 3]之间的数 模拟特征
    # 生成标签
    # 标签方程为 y = 0.5 * x **2+x+ 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数

    # 数据预处理，把x轴（特征）转成 多行 1列
    X = x.reshape(-1, 1)

    # 因为目前特征列只有1列，模型过于简单，会出现欠拟合的问题，我们增加1列 特征列，从而增加模型的复杂度.
    # 创建一个特征列
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5,X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 该函数作用：创建一个新特征列，进行水平拼接，该列是输入特征列的平方
    # 特征工程： polynomial
    # estmitor = LinearRegression()
    # 训练
    estmitor = Lasso(alpha=0.2) # Lasso 正则化系数 默认1.0
    estmitor.fit(X3, y)
    # print('权重：', estmitor.coef_)
    # print('截距：', estmitor.intercept_)

    # 预测
    y_predict = estmitor.predict(X3)

    # 评估
    print('均方误差：', mean_squared_error(y, y_predict))
    # print('RmsE：', root_mean_squared_error(y, y_predict))
    # print('平均绝对误差：', mean_absolute_error(y, y_predict))

    # 可视化
    plt.scatter(x, y)  # 绘制散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()

# L2正则化
def dm05_l2_regularization():
    # 指定随机种子 ，确保每次生成的数据集都是相同的
    np.random.seed(100)
    x = np.random.uniform(-3, 3, size=100)  # 随机生成100个[-3, 3]之间的数 模拟特征
    # 生成标签
    # 标签方程为 y = 0.5 * x **2+x+ 2 + 噪声
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 参1：均值，参2：标准差，参3：生成个数

    # 数据预处理，把x轴（特征）转成 多行 1列
    X = x.reshape(-1, 1)

    # 因为目前特征列只有1列，模型过于简单，会出现欠拟合的问题，我们增加1列 特征列，从而增加模型的复杂度.
    # 创建一个特征列
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5,X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 该函数作用：创建一个新特征列，进行水平拼接，该列是输入特征列的平方
    # 特征工程： polynomial
    # estmitor = LinearRegression()
    # 训练
    # estmitor = Lasso(alpha=0.2) # Lasso 正则化系数 默认1.0
    estmitor = Ridge(alpha=0.2)
    estmitor.fit(X3, y)
    # print('权重：', estmitor.coef_)
    # print('截距：', estmitor.intercept_)

    # 预测
    y_predict = estmitor.predict(X3)

    # 评估
    print('均方误差：', mean_squared_error(y, y_predict))
    # print('RmsE：', root_mean_squared_error(y, y_predict))
    # print('平均绝对误差：', mean_absolute_error(y, y_predict))

    # 可视化
    plt.scatter(x, y)  # 绘制散点图
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()

if __name__ == '__main__':
    # dm01_under_fitting()
    # dm02_just_fitting()
    # dm03_over_fitting()
    dm04_l1_regularization()