"""
案例:通过KNN算法实现 鸢尾花的分类操作.

回顾:机器学习项目的研发流程
    1.加载数据.
    2.数据的预处理.
    3.特征工程(提取，预处理...)
    4.模型训练.
    5.模型评估.
    6.模型预测.
"""

# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花测试集的.
import seaborn as sns # 可视化
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler      # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier    # KNN算法分类对象
from sklearn.metrics import accuracy_score            # 模型评估的，计算模型预测的准确率


# 定义函数，加载数据集
def dm01_load_iris():
    # 1.加载鸢尾花数据集
    iris_data = load_iris()
    print(f'数据集{iris_data.keys()}')
    print(f'数据集类型{type(iris_data)}')  # <class 'sklearn.utils._bunch.Bunch'>
    print(f'数据集特征{iris_data.data[0:5]}')
    print(f'数据集标签{iris_data.target[0:5]}')
    print(f'数据集标签名称{iris_data.target_names}')
    print(f'数据集特征名称{iris_data.feature_names}')
    # print(f'数据集描述{iris_data.DESCR}')

def dm02_show_iris():
    # 1.加载鸢尾花数据集
    iris_data = load_iris()
    # 2.数据可视化
    # 创建数据集，转成DataFrame
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    # 给数据集添加标签
    iris_df['label'] = iris_data.target
    # print(iris_df)
    # 绘制散点图
    sns.lmplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)',hue='label',fit_reg=False)
    plt.title('鸢尾花数据集')
    plt.tight_layout ()
    plt.show()

def dm03_split_train_test():
    # 1.加载鸢尾花数据集
    iris_data = load_iris()
    # 2.数据预处理
    # 返回值：训练集特征，测试集特征，训练集标签，测试集标签
   # train_test_split(数据集特征，数据集标签，测试集比例，随机种子)
    x_train,x_test,y_train,y_test =train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=23)
    # 3.打印切割后的结果
    print(f'训练集特征{x_train},个数{len(x_train)}')
    print(f'训练集标签{y_train},个数{len(y_train)}')
    print(f'测试集特征{x_test},个数{len(x_test)}')
    print(f'测试集标签{y_test},个数{len(y_test)}')

def dm04_iris_evaluate_test():
    iris_data=load_iris()
    # 数据预处理
    x_train,x_test,y_train,y_test =train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=23)
    # 特征工程
    # 思考1:特征提取:因为源数据只有4个特征列，且都是我们用的，所以这里无需做特征提取.
    # 思考2:特征预处理:因为源数据的4列特征差值不大，所以我们无需做特征预处理，但是，加入特征预处理会让我们的代码更完善，所以加入。
    # 创建数据标准化对象
    transfer = StandardScaler()
    # 对特征列进行标准化
    # fit_transform :兼具fit和transform的功能即:训练，转换.该函数适用于:第一次进行标准化的时候使用.
    x_train = transfer.fit_transform(x_train)
    # transform:只有转换.该函数适用于:重复进行标准化动作时使用，一般用于对测试集进行标准化.
    x_test = transfer.transform(x_test)
    # 模型训练
    estimate = KNeighborsClassifier(n_neighbors=3)
    estimate.fit(x_train,y_train) # 训练

    # 模型预测
    y_pre= estimate.predict(x_test)
    print(f'预测结果{y_pre}')
    # 场景2:对新的数据集(源数据150条 之外的数据)进行测试.
    my_data=[[7.8,2.1,3.9,1.6]]
    my_data = transfer.transform(my_data)
    # 模型预测
    y_pre_new = estimate.predict(my_data)
    print(f'预测结果{y_pre_new}')

    # 查看上述数据集，每种分类的预测概况
    my_pre_proba = estimate.predict_proba(my_data)
    print(f'预测概率{my_pre_proba}')

    # 模型评估
    # 方式1:直接评分，基于:训练集的特征 和 训练集的标签
    print(f'正确率(准确率)精确率:{estimate.score(x_test,y_test)}')

    # 方式2:基于测试集的标签和 预测结果 进行评分.
    print(f'准确率:{accuracy_score(y_test,y_pre)}')
if __name__ == '__main__':
    # dm01_load_iris()
    # dm02_show_iris()
    # dm03_split_train_test()
    dm04_iris_evaluate_test()