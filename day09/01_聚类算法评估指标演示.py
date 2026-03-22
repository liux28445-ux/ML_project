"""
案例:
    演示聚类算法的评估指标，即:SSE+肘部法，SC轮廓系数法，CH轮廓系数法.
聚类算法的评估指标:
    思路1:SSE+肘部法
        SSE:
            概述:
                所有簇的所有样本到该簇质心的 误差的平方和.
            特点:
                随着K值的增加，SSE值会逐渐减少
            目标:
                SSE值越小，代表簇内样本越聚集，内聚程度越高.
        肘部法:
            K值增大，SSE值会随之减小，下降梯度陡然变缓的得时候，那个K值，就是我们要的最佳值.
    思路2:SC轮廓系数
        考虑簇内 聚集程度，越小越好考虑簇外分离程度，越大越好.
    思路3:CH轮廓系数
        考虑簇内 -> 聚集程度，越小越好
        考虑簇外 -> 分离程度，越小越好.
        考虑K值 ->  K值越小，代表簇内样本越聚集，内聚程度越高.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
from sklearn.cluster import KMeans  # 引入KMeans
import matplotlib.pyplot as plt
from  sklearn.datasets import make_blobs  # 引入make_blobs,生成数据集
from sklearn.metrics import calinski_harabasz_score, silhouette_score  # 引入CH,作用：评估簇内样本的离散程度

def dm01_sse():
    sse_list= []
    x,y = make_blobs(
        n_samples=1000, # 数据集的样本数
        n_features=2,   # 特征数
        centers=[[-1,-1],[0,0],[1,1],[2,2]] # 簇的个数
        ,cluster_std=[0.4,0.2,0.2,0.2], # 标准差
        random_state=23
    )

    # 3.for训练遍历，获取到每个k值，计算其对应的 sse值，并添加到 sse_list列表中
    for k in range(1,100):
        # 创建KMeans模型,指定k值，迭代次数，随机种子
        kmeans = KMeans(n_clusters=k,n_init=100,random_state=23)
        kmeans.fit(x)
        # 此处不需要预测
        # 获取sse值
        sse_value = kmeans.inertia_
        sse_list.append(sse_value)
    # print(sse_list)
    # 4.绘制图像
    plt.figure(figsize=(20,10))
    plt.title('SSE')
    # 设置x轴刻度间隔
    plt.xticks(range(0,100,3))
    plt.xlabel('K')
    plt.ylabel('SSE')
    # 绘制图像,参数1为x轴数据，参数2为y轴数据
    plt.plot(range(1,100),sse_list)
    plt.show()

def dm02_sc():
    sc_list = []
    x, y = make_blobs(
        n_samples=1000,  # 数据集的样本数
        n_features=2,  # 特征数
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]]  # 簇的个数
        , cluster_std=[0.4, 0.2, 0.2, 0.2],  # 标准差
        random_state=23
    )

    # 3.for训练遍历，获取到每个k值，计算其对应的 sc值，并添加到 sc_list列表中
    # 至少两个簇
    for k in range(2, 100):
        # 创建KMeans模型,指定k值，迭代次数，随机种子
        kmeans = KMeans(n_clusters=k, n_init=100, random_state=23)
        kmeans.fit(x)
        # 模型预测
        y_predict = kmeans.predict(x)
        # 获取sc值
        sc_value = silhouette_score(x,y_predict)
        sc_list.append(sc_value)
    # print(sc_list)
    # 4.绘制图像
    plt.figure(figsize=(20, 10))
    plt.title('sc')
    # 设置x轴刻度间隔
    plt.xticks(range(0, 100, 3))
    plt.xlabel('K')
    plt.ylabel('sc')
    # 绘制图像,参数1为x轴数据，参数2为y轴数据
    plt.plot(range(2, 100), sc_list)
    plt.grid ()
    plt.show()

def dm03_ch():
    ch_list = []
    x, y = make_blobs(
        n_samples=1000,  # 数据集的样本数
        n_features=2,  # 特征数
        centers=[[-1, -1], [0, 0], [1, 1], [2, 2]]  # 簇的个数
        , cluster_std=[0.4, 0.2, 0.2, 0.2],  # 标准差
        random_state=23
    )

    # 3.for训练遍历，获取到每个k值，计算其对应的 ch值，并添加到 ch_list列表中
    # 至少两个簇
    for k in range(2, 100):
        # 创建KMeans模型,指定k值，迭代次数，随机种子
        kmeans = KMeans(n_clusters=k, n_init=100, random_state=23)
        kmeans.fit(x)
        # 模型预测
        y_predict = kmeans.predict(x)
        # 获取ch值
        ch_value = calinski_harabasz_score(x,y_predict)
        ch_list.append(ch_value)
    # print(ch_list)
    # 4.绘制图像
    plt.figure(figsize=(20, 10))
    plt.title('ch')
    # 设置x轴刻度间隔
    plt.xticks(range(0, 100, 3))
    plt.xlabel('K')
    plt.ylabel('ch')
    # 绘制图像,参数1为x轴数据，参数2为y轴数据
    plt.plot(range(2, 100), ch_list)
    plt.grid ()
    plt.show()
if __name__ == '__main__':
    # dm01_sse()
    # dm02_sc()
    dm03_ch()