"""
案例:
    演示线性回归 和 回归决策树(CART)对比.
细节:
    CART分类回归决策树，既可以做分类，也可以做回归，一般做:分类。
    做分类是采用 基尼值，做回归时采用 平方损失(类似于最小二乘)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor # 分类树
from sklearn.linear_model import LinearRegression # 线性回归


# 1.准备数据
x_train = np.array(list(range(1,11))).reshape(-1,1)
y_train = np.array([1.7, 2.8, 3.5, 4.5, 5.1, 5.9, 6.5, 7.2, 7.9, 8.5])

# 2.数据预处理
#3.特征工程
#4.模型训练
estimator1 = LinearRegression ()
estimator2 = DecisionTreeRegressor(max_depth=1) # CART分类树,max_depth=1,树最大深度为1
estimator3  = DecisionTreeRegressor(max_depth=3)

# 4.2  训练
estimator1.fit(x_train,y_train)
estimator2.fit(x_train,y_train)
estimator3.fit(x_train,y_train)

# 5.模型预测
# x_test = np.array(list(range(0.0,10.0)))
x_test = np.arange(0.0,10.0,0.1).reshape(-1,1)

# 具体预测动作
y_predict1 = estimator1.predict(x_test)
y_predict2 = estimator2.predict(x_test)
y_predict3 = estimator3.predict(x_test)

# 6.模型评估
print('线性回归:\n',estimator1.score(x_test,y_predict1))
print('CART分类树(max_depth=1):\n',estimator2.score(x_test,y_predict2))
print('CART分类树(max_depth=3):\n',estimator3.score(x_test,y_predict3))

# 7.可视化
# plt.figure()
plt.scatter(x_train,y_train)
# 以预测值（线性回归，CART分类树）绘制折线图
plt.plot(x_test,y_predict1,color='r',label='LinearRegression')
plt.plot(x_test,y_predict2,color='g',label='DecisionTreeClassifier(max_depth=1)')
plt.plot(x_test,y_predict3,color='b',label='DecisionTreeClassifier(max_depth=3)')
plt.legend()
# 添加x轴和y轴标签
plt.xlabel('data')
plt.ylabel('target')
plt.title('LinearRegression VS DecisionTreeClassifier')
plt.savefig('./data/01_线性回归和回归决策树对比.png')
plt.show()