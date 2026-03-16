"""
线性回归介绍(Linear Regressor):
    概述/目的:
        用线性公式 来描述 多个自变量(特征) 和1个因变量(标签)之间 关系的，对其关系进行建模，基于特征 预测 标签.
        线性回归属于:有监督学习，即:有特征，有标签，且标签是连续的.
    分类:
    一元线性回归:1个特征列+1个标签列
    多元线性回归:多个特征列+1个标签列
    公式:
        一元线性回归:
            y=kx+b => wx+b
            k:数学中叫斜率，在机器学习中Weight(权重)，简称:w, b:数学中叫截距，在机器学习中Bias(偏置)，简称:b
        多元线性回归:
            y=w1x1+w2x2+w3x3+...+wn@xn+b
            =W的转置*x+b
    误差 = 预测值 - 真实值
    损失函数(Loss Function,也叫成本函数，代价函数，目标函数，Cost Function):
        用于描述 每个样本点 和 其预测值之间关系的，让损失函数最小，就是让误差和小，线性回归效率，评估就越高.
    如何让损失函数最小：
     思路1：正规方程法。
     思路2：梯度下降法

     损失函数分类:
        最小二乘:每个样本点误差的平方和
        MSE(Mean Square Error，均方误差):每个样本点误差的平方和/样本个数
        RMSE(Root Mean Square Error，均方根误差):均方误差开平方根
        MAE(Mean Absolute Error，均绝对误差):每个样本点误差的绝对值和/样本个数
"""
# 导入
from sklearn.linear_model import LinearRegression

x_train = [[160], [170], [176], [180], [182]] # 训练集特征
y_train = [65.2, 74.4, 80.2, 85.9, 90]        # 训练集标签

x_test = [[178]]                              # 测试集特征

# 模型训练
# 创建模型对象
estimator = LinearRegression()
estimator.fit(x_train, y_train)

print(f'权重{estimator.coef_}')
print(f'偏置{estimator.intercept_}')
pre = estimator.predict(x_test)
print(f'预测结果{pre}')