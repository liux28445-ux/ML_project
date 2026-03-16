import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 1. 加载乳腺癌数据集（二分类任务）
data = load_breast_cancer()
X, y = data.data, data.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建模型
# 关键参数：
# criterion='gini': 使用基尼系数
# max_depth=3: 预剪枝，限制树只有3层，防止过拟合
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. 评估模型
print(f"训练集准确率: {clf.score(X_train, y_train):.2f}")
print(f"测试集准确率: {clf.score(X_test, y_test):.2f}")

# 5. 可视化这棵树 (最酷的部分)
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True,
          rounded=True,
          fontsize=12)
plt.show()