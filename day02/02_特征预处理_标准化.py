"""案例:演示特征预处理之归一化操作
回顾:特征工程的目的 和 步骤
目的:
利用专业的背景知识 和 技巧处理数据，用于提升模型的性能.
步骤:
1.特征提取.
2.特征预处理(归一化，标准化)
3.特征降维.
4.特征选择.
5.特征组合.
特征预处理之 归一化介绍:
目的:
防止因为量纲(单位)问题，导致特征列的方差值相差较大影响模型的最终结果.
所以通过公式把各列值 映射到 均值为0，标准差为1的 正态分布序列.
公式:
x'=(当前值-该列平均值值)/该列的标准差 σ
应用场景：
    适用于 大数据集 的处理
"""

from sklearn.preprocessing import StandardScaler  # 对象

x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 创建对象
transfer = StandardScaler()

# 训练数据
x_train_new = transfer.fit_transform(x_train)

print(x_train_new)

# 打印数据的均值和方差
print(f' 数据集的均值为:{transfer.mean_}')
print(f' 数据集的方差为:{transfer.var_}')