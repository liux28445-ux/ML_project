"""
    案例:演示KNN算法识别图片，即:手写数字识别案例.
    介绍:
    每张图片都是由28*28 像素组成的，即:我们的CsV文件中每一行都有 784个像素点，表示图片(每个像素)的颜色.
    最终构成图像.

"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter  # 去重统计
# 忽略警告
import warnings # 参1：忽略警告 参2 ：忽略的模块
warnings.filterwarnings("ignore",module="sklearn")


# 定义函数，接受用户传入的索引
def show_digit(idx):
    # 读取数据集
    data = pd.read_csv("data/手写数字识别.csv")
    # print(data.describe())
    if idx < 0 or idx > len(data)-1:
        print("索引值超出范围")
        return
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # 查看用户传入的索引对应的图片
    print("图片对应的数字为:", y[idx])

    # 查看下 x 的形状
    # print(x.iloc[idx].shape)
    x= x.iloc[idx].values.reshape(28, 28)

    plt.imshow(x, cmap="gray") # 灰度图
    plt.show()

# 定义函数，训练模型，保存模型
def train_model():
    # 读取数据集
    data = pd.read_csv("data/手写数字识别.csv")
    # 数据预处理
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    # 对特征列归一化
    x = x / 255
    # 划分数据集,参5:参考y轴进行抽取，保持标签比例
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22,stratify= y)

    # 模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 模型评估
    print("准确率:", estimator.score(x_test, y_test))

    # 保存模型
    joblib.dump(estimator, "my_model/手写数字识别.pkl")

# 模型测试
def use_model():
    # 读取数据集
    img = plt.imread("data/demo.png")
    # plt.imshow(img, cmap="gray")
    # plt.show()

    # 加载模型
    estimator = joblib.load("my_model/手写数字识别.pkl")

    # 模型预测
    # print(img.shape)
    # print(img.reshape(1,784))
    # print(img.reshape(1,-1)) # 语法糖,-1就是能转多少转多少

    # 查看数据集转换,记得归一化（因为训练模型的时候也使用了归一化
    img = img.reshape(1, -1) # /255 可能会失败

    #  模型预测
    y_pre = estimator.predict(img)
    print("预测结果为:", y_pre)
if __name__ == '__main__':
    # show_digit(1)
    # train_model()
    use_model()
