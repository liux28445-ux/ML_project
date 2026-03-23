"""
案例:
    演示混淆矩阵和 精确率，召回率，F1值.
回顾:逻辑回归
    概述:
        属于有监督学习，即:有特征，有标签，且标签是离散的.适用于二分类.
    评估:
        精确率，召回率，F1值
混淆矩阵:
    概述:
        用来描述 真实值 和 预测值之问关系的.
    图解:
                 真实标签(反例)  预测标签(反例)
    真实标签(正例)  真正例(TP)     伪正例(FP)
    预测标签(正例)  真反例TN)      伪反例(FN)

    单词:
        True:真, False:假(伪)
        Positive:正例
        Negative:反例
结论:
1.模拟使用 分类少的 充当 正例.
2.精确率=真正例在预测正例中的占比，即:tp/(tp+fp)
3.召回率=真正例在真正例中的占比，即:tp/(tp+fn)
4.F1值=2*(精确率* 召回率)/(精确率+召回率)
"""

import pandas as pd
# confusion_matrix:混淆矩阵 precision_score:精确率 recall_score:召回率 f1_score:F1值
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
# 需求:已知有10个样本，6个恶性肿瘤(正例)，4个良性肿瘤(反例).
# #模型A预测结果为:预测对了3个恶性肿瘤，预测对了4个良性肿瘤
# #模型B预测结果为:预测对了6个恶性肿瘤，预测对了1个良性肿瘤
# 定义变量
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

y_pre_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性']
y_pre_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '恶性', '恶性']

lable = ['恶性', '良性']
df_lable = ['恶性(正例）', '良性（反例）']
# 混淆矩阵
cma = confusion_matrix(y_train, y_pre_A, labels=lable)
print(f'混淆矩阵为:{cma}')
print(f'混淆矩阵为:{pd.DataFrame(cma, index=df_lable, columns=df_lable)}')
print(f'精确率:{precision_score(y_train, y_pre_A,pos_label='恶性')}') # pos_label='恶性' 代表正例
print(f'召回率:{recall_score(y_train, y_pre_A,pos_label='恶性')}')
print(f'F1值:{f1_score(y_train, y_pre_A,pos_label='恶性')}')