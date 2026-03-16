"""
     演示 CART 分类回归决策树 的分类功能
"""
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report # 评价报告
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree # 绘制树形图

# 1.加载数据
data = pd.read_csv('./data/train.csv')
# 2，数据预处理
# 2.1 提取特征和标签
x = data[['Pclass', 'Age', 'Sex']]
y= data['Survived']

# print(x.head(5))

# 2.2 缺失值处理
# x['Age'].fillna(x['Age'].mean(),inplace=True) # 会报警告，但是可以用
x = x.copy() # 拷贝数据，不写也行
x['Age'] = x['Age'].fillna(x['Age'].mean())# 会报告，但是可以用
# 2.3 针对Sex列经行热编码one- hot
x = pd.get_dummies(x,columns=['Sex'])
# 2.4 划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)

# 3.特征工程

# 4.模型训练
# 参数1:M=max_depth,树最大深度
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train,y_train)
# 5.模型预测
y_predict = estimator.predict(x_test)
print('预测结果:\n',y_predict)
# 6.模型评估

print('准确率:\n',estimator.score(x_test,y_test))
print('分类报告:\n',classification_report(y_test,y_predict))

# 7.可视化
plt.figure(figsize=(150,100))
# 参数1:estimator,模型参数2：颜色填充，参数3：最大深度
plot_tree(estimator,filled=True,max_depth=10)
plt.savefig('./data/titanic_tree2.png')
plt.show()