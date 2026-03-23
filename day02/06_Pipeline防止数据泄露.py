from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# 1. 加载数据并进行基础划分
X, y = load_iris(return_X_y=True)
# 预留出最终的测试集，不参与调参
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 搭建流水线 (Pipeline)：先标准化，再送入KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 第一步：标准化
    ('knn', KNeighborsClassifier()) # 第二步：KNN分类器
])

# 3. 设置网格搜索的参数范围 (注意参数名的写法：步骤名__参数名)
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],       # 搜索最优的 K 值
    'knn__weights': ['uniform', 'distance'],    # 搜索最优的权重计算方式
    'knn__p': [1, 2]                            # 搜索距离公式：1为曼哈顿，2为欧氏
}

# 4. 初始化网格搜索，结合 5折交叉验证 (cv=5)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# 5. 执行自动搜索与训练
print("开始网格搜索与交叉验证...")
grid_search.fit(X_train, y_train)

# 6. 查看结果
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"交叉验证最高得分: {grid_search.best_score_:.4f}")

# 7. 使用自动找到的最佳模型在未知测试集上进行最终评估
best_model = grid_search.best_estimator_
final_score = best_model.score(X_test, y_test)
print(f"最终测试集准确率: {final_score:.4f}")