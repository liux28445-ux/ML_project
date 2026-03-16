"""
案例:演示网格搜索和交叉验证.

交叉验证解释:
    原理:
        把数据分成n份，例如分成:4份
        第1次:把第1份数据作为验证集(测试集)，其它作为训练集，训练模型，模型预测，获取:准确率准确率1
        第2次:把第2份数据作为验证集(测试集)，其它作为训练集，训练模型，模型预测，获取:准确率  准确率2
        第3次:把第3份数据作为验证集(测试集)，其它作为训练集，训练模型，模型预测，获取:准确率准确率3
        第4次:把第份数据作为 验证集(测试集)，其它作为训练集，训练模型，模型预测，获取:准确率准确率4

        然后计算上述的4次准确率的平均值，作为:模型最终的准确率。

        假设第4次最好(准确率最高)，则:用全部数据(训练集+ 测试集)训练模型，再次用测试集对模型测试.

网格搜索:
    目的/作用:
        寻找最优超参数.
    原理:
        接超参可能出现的值，收然后针对于 超参的每个值进行 交叉验证，获取到 最优超参组合.：
    超参数:
        需要用户手动录入的数据，不同的超参(组合)，可能会影响模型的最终评测结果.
    大白话解释:
        网格搜索+交叉验证，本质上指的是GridSearchCV这个API，它会帮我们寻找最优超参
"""

from sklearn.datasets import load_iris                  # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split,GridSearchCV  # 分割训练集和测试集的,寻找最优超参的（网格搜索+交叉验证）
from sklearn.preprocessing import StandardScaler      # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier    # KNN算法分类对象
from sklearn.metrics import accuracy_score            # 模型评估的，计算模型预测的准确率


# 1.加载数据集
iris = load_iris()

# 2.数据预处理
x_train,x_test,y_train,y_test =train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)

# 3.数据标准化
transfer = StandardScaler()

# 对训练集与测试特征进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4.KNN算法模型训练
# 创建模型对象
estimator = KNeighborsClassifier()

# 定义字典，存储超参数和超参数的取值
param_dict = {"n_neighbors":[x for x in range(1,11)]}
# 创建网格搜索对象 -> GridSearchCV寻找最优超参使用网格搜索+交叉验证
# 参1: 模型对象，参2: 超参数的取值字典，参数名：参数值列 表参3: 交叉验证的次数
estimator = GridSearchCV(estimator=estimator,param_grid=param_dict,cv=4)
# 具体模型训练
estimator.fit(x_train,y_train)
# 打印最优超参组合
print("最优超参组合:",estimator.best_params_)    # {'n_neighbors': 3}
print("最优超参的准确率:",estimator.best_score_)  # 0.9666666666666668
print(f'最优的估计器对象:{estimator.best_estimator_}') # KNeighborsClassifier(n_neighbors=3)
print(f'交叉验证的准确率:{estimator.cv_results_}')

# 模型评估
estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train,y_train)

y_pre = estimator.predict(x_test)
print("模型预测的准确率:",accuracy_score(y_test,y_pre))