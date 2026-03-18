"""
案例:
    演示 Boosting思想之 GBDT(Gradient Boosting Decision Tree,梯度提升树)处理泰坦尼克号数据集.
GBDT 梯度提升树解释:
    概述:
    通过拟合负梯度 来获取一个强学习器
    流程:
    1.采用所有目标值的均值 作为第1个弱学习器的预测值.
    2.目标值-预测值=负梯度(残差)，该(列值作为第2个弱学习器的目标值.
    3.针对于第1个弱学习器，依次计算每个分割点的 最小平方和，找到最佳 分割点，至此:第1个弱学习器搭建完毕。
    4.把上述的分割点带入第2个弱学习器，计算它的预测值=以此分割点为界，目标值的均值，即为该部分数据的 预测值。
    5.计算第2个弱学习器的负梯度，最佳分割点，至此:第2个弱学习器搭建完毕.
    6.以此类推，直至程序结束.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier # 梯度提升数分类器
from sklearn.tree import DecisionTreeClassifier # 决策分类器
from  sklearn.metrics import classification_report # 模型评估
from sklearn.model_selection import GridSearchCV # 网格搜索

df = pd.read_csv('./data/train.csv')

x = df[['Pclass', 'Age', 'Sex']].copy()
y = df['Survived'].copy()

# 处理age的缺失值，用该列的均值填充
x['Age']=x['Age'] .fillna(x['Age'].mean())

# 热编码处理
x = pd.get_dummies(x)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 创建模型对象
estimator = DecisionTreeClassifier ()

estimator.fit(x_train, y_train)
y_predict = estimator.predict(x_test)
print(f'单个决策树对象{y_predict}')
print(f'单个决策树对象分类报告\n{classification_report(y_test, y_predict)}')

# 梯度提升树预测结果
estimator2 = GradientBoostingClassifier()
estimator2.fit(x_train, y_train)
y_predict2 = estimator2.predict(x_test)
print(f'梯度提升树预测结果{y_predict2}')
print(f'梯度提升树预测结果分类报告\n{classification_report(y_test, y_predict2)}')

# 针对GBDT模型，寻找参数
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],       # 弱学习器的数量
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],# 学习率
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]     # 树的深度
}
estimator3 = GradientBoostingClassifier()
estimator4 = GridSearchCV(estimator3, param_grid, cv=5)
estimator3.fit(x_train, y_train)
estimator4.fit(x_train, y_train)
print(estimator4.predict(x_test))
print(estimator4.score(x_test, y_test))
print(f'梯度提升树最优参数预测结果分类报告\n{classification_report(y_test, estimator4.predict(x_test))}')