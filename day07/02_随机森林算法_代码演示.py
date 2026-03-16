"""
突例:
演示 集成学习之 Bagging思想 随机森林算法 代码.
集成学习:
概述:
把多个弱学习器 组成1个强学习器的过程  集成学习.
思想:
Bagging思想:
1.有放回的随机抽样.
2.平权投票.
3.可以并行执行.
Boosting思想:
1.每次训练都会使用全部样本.
2.加权投票预测正确:权重降低，预测错误:权重增加.
3.只能串行执行.
Bagging思想代表:
随机森林算法.
随机森林算法:
1.每个弱学习器都是 CART树(必须是二叉树)
2.有放回的随机抽样，平权投票，并行执行.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from  sklearn.tree import DecisionTreeClassifier # 引入 决策树分类器
from sklearn.ensemble import  RandomForestClassifier # 引入 随机森林分类器
from sklearn.model_selection import  GridSearchCV # 引入 网格搜索

# 1. 加载数据
df = pd.read_csv("./data/train.csv")
# 2，数据的预处理
x =df[['Pclass','Sex','Age']].copy()
y=df['Survived']

# 处理空值数据
x['Age'] = x['Age'].fillna(x['Age'].mean())
#热编码处理
x=pd.get_dummies(x)
# 数据集的划分
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)

# 3. 特征工程,此处略

# 4. 模型训练
# 创建决策树对象
estimator1  = DecisionTreeClassifier()
estimator1.fit(x_train,y_train)
# 5. 模型预测
pre1 = estimator1.predict(x_test)
# print(f'单一决策树预测结果为:{pre1}')
# 6. 模型评估
print(f'单一决策树准确率:{estimator1.score(x_test,y_test)}')

# 创建随机森林对象 -- 默认参数
estimator2 = RandomForestClassifier() # n_estimators=100 -- 树的数量, criterion='gini' -- 评价标准,
estimator2.fit(x_train,y_train)
pre2 = estimator2.predict(x_test)
print(f'随机森林准确率:{estimator2.score(x_test,y_test)}')

# 随机森林算法 -- 参数调优，网格搜索
estimator3 = RandomForestClassifier()
params = {
    'n_estimators': [100,200,300,400,500,1000],
    'max_depth': [3,5,7,9]
}
gs = GridSearchCV(estimator3,params,cv=3)
gs.fit(x_train,y_train)
pre3=gs.predict(x_test)
print(f'随机森林准确率:{gs.score(x_test,y_test)}')
print(f'最佳参数:{gs.best_params_}')
