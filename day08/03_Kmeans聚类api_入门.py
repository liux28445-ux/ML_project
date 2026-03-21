"""
案例:
    演示KMeans 聚类算法 入门案例.
Kmeans简介:
    它属于无监督学习，即:有特征，无标签，根据样本问的相似性进行划分.
    所谓的相似性 可以理解为就是 距离，例如:欧式距离，曼哈顿(城市街区)距离，切比雪天距离，闵式距离...
    一般大厂，项日初期在没有先备知识(标签)，可能会用
"""

from sklearn.cluster import KMeans   # 聚类的API，采用指定 质心 来分簇
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # 默认按照高斯分布（正态分布）生成数据集，只需要指定均值，标准差
from sklearn.metrics import calinski_harabasz_score # 评价指标, 用来评价簇的个数

# 生成数据集
# 参1(n_samples)：数据集的样本数，参2(n_features)：特征数，参数3(centers)：簇的个数，参数4(cluster_std)：标准差，参数5(random_state)：随机数种子
x,y = make_blobs(n_samples=1000,
                 n_features=2,
                 centers=[[-1,-1],[0,0],[1,1],[2,2]],
                 cluster_std=1.0,
                 random_state=23)

# plt.scatter(x[:,0],x[:,1])
# plt.show()

# 创建模型
# 参数1(n_clusters)：指定簇的个数，参数2(random_state)：随机数种子
estimator = KMeans(n_clusters=4,random_state=23)
y_pre = estimator.fit_predict(x)

plt.scatter(x[:,0],x[:,1],c=y_pre)
plt.show()

print(f'评测指标{calinski_harabasz_score(x,y_pre)}')