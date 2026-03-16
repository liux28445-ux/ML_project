"""
案例:
    通过逻辑回归算法，针对于电信用户数据建模，进行流失预测分析.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 切割数据集
from sklearn.linear_model import LogisticRegression
# 准确率，精确率，召回率，F1值，分类评估报告
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

def dm01_data_preprocess():
    churn_df = pd.read_csv('./data/churn.csv')
    # print(churn_df.info())

    # 因为Churn 和gender 是字符串，所以要进行one-hot编码（热编码处理）
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])

    # 查看数据
    # print(churn_df.head(5))

    # 删除one-hot处理后，冗余的列
    churn_df.drop(['Churn_False.', 'gender_Male'], axis=1, inplace=True)

    # 修改列名，将Churn_Yes-> flag,充当标签列
    churn_df.rename(columns={'Churn_True.': 'flag'}, inplace=True)

def dm02_data_visualization():
   # 读取 数据
    churn_df = pd.read_csv('./data/churn.csv')
   # 数据处理
    churn_df = pd.get_dummies(churn_df, columns=['Churn', 'gender'])
   # 删除one-hot处理后，冗余的列
    churn_df.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True)
   # 修改列名，将Churn_Yes-> flag,充当标签列
    churn_df.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    """
    Index(['Partner_att', 'Dependents_att', 'landline', 'internet_att',
       'internet_other', 'StreamingTV', 'StreamingMovies', 'Contract_Month',
       'Contract_1YR', 'PaymentBank', 'PaymentCreditcard', 'PaymentElectronic',
       'MonthlyCharges', 'TotalCharges', 'flag', 'gender_Female'],
      dtype='object')
    """
    # 查看列名
    # print(churn_df.columns)
    # 绘制数据可视化,参数1：数据源，参数2：x轴，参数3：hue表示分组，根据分组进行绘制
    sns.countplot(data=churn_df, x='Contract_Month',hue='flag')
    plt.show()

def dm03_logistic_regression():
    # 1.加载数据
    churn_df = pd.read_csv('./data/churn.csv')

    # 数据的预处理
    # one-hot编码
    churn_df = pd.get_dummies(churn_df, columns=['Churn_Yes', 'gender'])
    # 删除one-hot处理后，冗余的列
    churn_df.drop(['Churn_No', 'gender_Male'],axis=  1,inplace= True)
    # 修改列名，将Churn_Yes-> flag,充当标签列
    churn_df.rename(columns={'Churn_Yes': 'flag'},inplace= True)
    # 数据切割
    #x的特征列：月度会员 ，是否有互联网服务，是否电子支付
    x = churn_df[['Contract_Month','internet_other','PaymentElectronic']]
    y = churn_df['flag']

    # 数据切割
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    # 特征工程，暂不处理

    #模型训练
    # 逻辑回归
    estimator = LogisticRegression()
    # 模型训练
    estimator.fit(x_train,y_train)

    # 模型预测
    y_predict = estimator.predict(x_test)
    print(f'预测结果{y_predict}')

    # 模型评估
    print(f'准确率{accuracy_score(y_test,y_predict)}')
    print(f'精确率{precision_score(y_test,y_predict)}')
    print(f'召回率{recall_score(y_test,y_predict)}')
    print(f'F1值{f1_score(y_test,y_predict)}')
    print(f'分类评估报告{classification_report(y_test,y_predict)}')

if __name__ == '__main__':
    # dm01_data_preprocess()
    # dm02_data_visualization()
    dm03_logistic_regression()