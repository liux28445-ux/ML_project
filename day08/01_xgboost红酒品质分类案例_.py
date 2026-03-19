"""
案例:
    通过XGBo0st 极限梯度提升树完成红酒品质分类案例.
回顾:XGB00st 极限梯度提升树
    慨述:
        Extreme Gradient Boosting Tree,底层采用打分函数决定是否分支.
    原理:
        Gain值=分枝前的打分-(分支后左子树打分+分支后右子树打分)如果Gain值>0，考虑分枝，否则:不考虑分枝.
"""

import joblib                                           # 保存和加载模型
import numpy as np
import pandas as pd
import xgboost as xgb                                   # 极限梯度提升树
from collections import Counter                         # 统计数据
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report       # 模型（分类）评估报告
from sklearn.model_selection import StratifiedKFold     #  分层K折交叉验证，类似于 网格搜索时 cv = 折数
from sklearn.utils import class_weight



# 1.定义函数,进行数据切割
def dm01_data_split():
    df = pd.read_csv('./data/红酒品质分类.csv')

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1] - 3

    # print(f'数据集的标签类别分布:{Counter(y)}')

    # 切分,参1:数据集,参2:测试集所占比例,参3:随机数种子,参4:是否分层
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22,stratify= y)

    # 把训练集特征和标签拼接一起
    pd.concat([x_train,y_train],axis=1).to_csv('./data/红酒品质分类_train.csv',index=False)
    pd.concat([x_test,y_test],axis=1).to_csv('./data/红酒品质分类_test.csv',index=False)

def dm02_train_model():
    train_data = pd.read_csv('./data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./data/红酒品质分类_test.csv')

    x_train = train_data.iloc[:, :-1] # 训练集特征
    y_train = train_data.iloc[:, -1] # 训练集标签
    estimator = xgb.XGBClassifier(
        n_estimators=100,  # (树)弱学习器的数量
        max_depth=3,  # 树的最大深度
        # learning_rate=0.1,  # 学习率
        objective='multi:softmax',  # 目标函数
        # num_class=3,  # 类别数
        # n_jobs=-1,  # 使用的CPU进程数
        random_state=22 # 随机数种子
    )
    # 计算样本权重,作用:使模型在训练时，样本权重更平衡，从而提高模型的泛化性能
    # 参1:权重计算方式,参2:标签
    class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    estimator.fit(x_train, y_train)
    print(f'训练集的预测结果:{estimator.predict(x_train)}')
    print(f'训练集的准确率:{estimator.score(x_train, y_train)}')
    joblib.dump(estimator, './data/红酒品质分类_model.pkl')

def dm03_use_model():
    train_data = pd.read_csv('./data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./data/红酒品质分类_test.csv')

    x_train = train_data.iloc[:, :-1]  # 训练集特征
    y_train = train_data.iloc[:, -1]  # 训练集标签
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    estimator = joblib.load('./data/红酒品质分类_model.pkl')

    # 创建网格搜索 + 交叉验证(结合分层采样数据)
    param_dict = {
        'max_depth': [4, 5, 6, 7, 8,],
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001]
    }
    # 创建分层采样对象
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

    # 创建网格搜索对象
    gs_estimator = GridSearchCV(estimator, param_grid=param_dict, cv=skf)
    gs_estimator.fit(x_train, y_train)
    print(f'训练集的预测结果:{gs_estimator.predict(x_train)}')
    print(f'训练集的准确率:{gs_estimator.score(x_train, y_train)}')

    y_pre = gs_estimator.predict(x_test)

    print(f'测试集的预测结果:{y_pre}')
    print(f'测试集的准确率:{gs_estimator.score(x_test, y_test)}')
    print(f'最优参数:{gs_estimator.best_params_}')
    print(f'最优估计器:{gs_estimator.best_estimator_}')
if __name__ == '__main__':
    # dm01_data_split()
    # dm02_train_model()
    dm03_use_model()