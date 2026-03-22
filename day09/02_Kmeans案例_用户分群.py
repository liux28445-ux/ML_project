"""
案例:
    基于用户的年收入 和 消费质数，根据用户的相似性进行聚类.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.cluster import KMeans # 引入KMeans算法
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score,silhouette_score # 引入CH,作用：评估簇内样本的离散程度
import pandas as pd

#1.定义函数，找聚类的质心数(k值)
def dm01_find_k():
    df = pd.read_csv('./data/customers.csv')

    # 定义ssc_list,sc_list，记录不同k值 评估效果
    sse_list = []  # sse:只考虑簇内样本，越小越好
    sc_list = []   # sc:考虑簇内样本和簇间样本，越大越好
    # 抽取特征
    x = df.iloc[:,3:5]
    # 定义for训练，获取不同k值，对应的sse,sc，
    for k in range(2,20):
        # 创建KMeans模型,指定k值，迭代次数，随机种子
        kmeans = KMeans(n_clusters=k,n_init=100,random_state=23)

        # 模型训练
        kmeans.fit(x)
        # 模型预测
        y_predict = kmeans.predict(x)
        # 获取sse值
        sse_value = kmeans.inertia_
        sse_list.append(sse_value)
        # 获取sc值
        sc_value = silhouette_score(x,y_predict)
        sc_list.append(sc_value)

    # 绘制图像
    plt.figure(figsize=(20,10))
    # # 设置x轴刻度间隔
    # plt.xticks(range(0,20,1))
    plt.plot(range(2,20),sse_list,label='SSE')
    plt.show()
    plt.figure(figsize=(20,10))
    plt.plot(range(2,20),sc_list,label='SC')
    # plt.xticks(range(0,20,1))
    plt.show()
    # 结论：质心数k=5，sc最大，越小越好，越接近1越好，越接近0越不好


#2.定义函数，实现:模型训练，模型预测，模型评估
def dm02_train_predict_evaluate():
    df = pd.read_csv('./data/customers.csv')
    x= df.iloc[:,3:5]

    # 创建KMeans模型,指定k值，迭代次数，随机种子
    # k=5是刚才找到的质心数k值，通过 SSE+肘部法 , SC轮廓系数法
    estimator = KMeans(n_clusters=5,n_init=100,random_state=23)
    estimator.fit(x)
    y_predict = estimator.predict(x)
    # 绘制 5个簇 的样本点
    plt.scatter(x.values[y_predict == 0,0],x.values[y_predict == 0,1]) # 0号簇
    plt.scatter(x.values[y_predict == 1,0],x.values[y_predict == 1,1]) # 1号簇
    plt.scatter(x.values[y_predict == 2,0],x.values[y_predict == 2,1]) # 2号簇
    plt.scatter(x.values[y_predict == 3,0],x.values[y_predict == 3,1]) # 3号簇
    plt.scatter(x.values[y_predict == 4,0],x.values[y_predict == 4,1]) # 4号簇
    # plt.show()
    # 绘制 5个簇 的质心
    plt.scatter(estimator.cluster_centers_[:,0],estimator.cluster_centers_[:,1])
    # plt.title('Centers')
    plt.show()

#3.测试
if __name__ == '__main__':
    # dm01_find_k()
    dm02_train_predict_evaluate()