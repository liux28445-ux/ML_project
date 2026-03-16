"""
算法介绍(K近邻)，K近邻算法
    原理:
        基于 欧式距离(或者其它距离计算方式)计算 测试集 和 每个训练集之间的距离，然后根据距离升序排列，找到最近的K个样本.
        基于K个样本投票，票数多的就作为最终预测结果 分类问题。
        基于K个样木`计算平均值，作为最终预测结果   回归问题.
    实现思路:
         1.分类问题
            适用于:有特征，有标签，且标签是不连续的(离散的)
        2.回归问题.
            适用于:有特征，有标签，且标签是连续的.
    KNN算法，分类问题思路如下:
        1.计算 测试集 和 每个训练的样本 之间的 距离.
        2.基于距离进行升序排列.
        3.找到最近的K个样本.
        4.K个样本进行投票.
        5.票数多的结果，作为最终地预测结果.
    代码实现思路：
        1.导包
        2.准备数据集（测试 |训练）
        3.创建(KNN 分类模型)模型对象
        4.模型训练.
        5.模型预测.
"""

from sklearn.neighbors import KNeighborsClassifier

# train: 训练集
# test: 测试集
# neighbors: K值
#estimator: 模型对象,估计器，也可以用model接受

x_train = [[0],[1],[2],[3]]
y_train = [0,0,1,1]

x_test=[[5]]

estimator = KNeighborsClassifier(n_neighbors=3) # 创建模型对象 , n_neighbors:K值

estimator.fit(x_train,y_train) # 模型训练

y_pre = estimator.predict(x_test)
print(y_pre)