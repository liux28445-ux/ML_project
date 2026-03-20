"""
案例:
    演示通过朴素贝叶斯算法实现 商品评论情感分析，即:好评，差评...
朴素贝叶斯介绍:
    概述:
        贝叶斯:仅仅依赖 概率 就可以进行分类的一种机器学习算法.朴素:不考虑特征之间的关联性，即:特征间都是相互独立的.原始: P(AB)= P(A)* P(B|A) = P(B) * P(A|B)加入朴素后:P(AB)=P(A)*P(B)
    细节:
        因为我们分词要用到jieba分词器，记得先装一下，例如:pip install jieba
"""


import pandas as pd         # 数据处理
import numpy as np
import jieba             # 分词包
from  sklearn.feature_extraction.text import CountVectorizer   # 词频统计，把评论内容 转成 词频矩阵
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB                  # 朴素贝叶斯对象

df = pd.read_csv('./data/书籍评价.csv',encoding='gbk')

# 添加labels列，充当标签列，好评 - 1，差评 - 0
df['labels'] =np.where(df['评价'].str.contains('好评'),1,0)

# 抽取labels列作为标签
y = df['labels']

# 演示 jieba 分词
# print(jieba.lcut('小明骑车，一把把把把把住了'))

# 对用户的评论信息，做切词
# comment_list = [jieba.lcut(line) for line in df['内容']]
# 创建词频统计对象
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
# print(comment_list)

# 演示字符串的join()函数用法.
# my_list = ['aa','bb','cc']
# print(','.join(my_list))

# 加载停用词列表，即：里面记录的词，不需要参与模型训练，预测，例如:我们、的、是、了、啊、的
with open('./data/stopwords.txt',encoding='utf-8') as src_f:
    # 把停用词列表，读取到列表中,一次读取所有行
    stopword_list = src_f.readlines()
    # 删除换行符
    stopword_list = [line.strip() for line in stopword_list]
    #对停用词去重
    stopword_list = list(set(stopword_list))

# 创建向量化对象，从 评论切词列表(comment_list)中删除 停用词，并且统计词频(单词矩阵).
transfer = CountVectorizer(stop_words=stopword_list)
# 调用 fit_transform()方法，把评论列表转换成单词矩阵
x = transfer.fit_transform(comment_list).toarray()

#看一下我们13条评论，切词，且删除 停用词后，一共剩下多少个词了.
# print(transfer.get_feature_names_out()) #37个词，即:13条评论，切词，且删除 停用词后，一共剩下多少个词了.

# 创建训练集和测试集
# 因为就 13条数据，我们把前10条当训练集，后三条当测试集。
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

# 创建朴素贝叶斯对象
estimator = MultinomialNB()
# 调用 fit()方法，训练模型
estimator.fit(x_train,y_train)

# 预测
y_predict = estimator.predict(x_test)
print(f'预测结果:{y_predict}')
print(f'准确率{accuracy_score(y_test,y_predict)}')