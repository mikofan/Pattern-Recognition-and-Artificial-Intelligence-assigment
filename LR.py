'''
仅用于一元线性回归的模块

输入：

    文件名

输出：

    1 数据集的样本分布结果（可视化）

    2 线性回归拟合结果（可视化）

    3 预测在面积大小为3.1415的城市开一家餐厅的预计利润
'''

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
#负号正常显示
plt.rcParams['axes.unicode_minus'] = False

def start_LR(fileName):
    #读取文件，因为本txt没有列名，所以header=None
    dataset = pd.read_csv(fileName, header=None)
    
    #提取特征，0列为面积，1列为利润，X、Y为n行1列的列表，需要用函数转换
    space_feature = dataset[0]
    income_feature = dataset[1]
    X = np.reshape(space_feature.values, (-1, 1))
    Y = np.reshape(income_feature.values, (-1, 1))
    
    #画出数据集
    print("This is the map of datasets of squares and incomes:\n")
    plt.scatter(X, Y, color='red')
    plt.xlabel('城市面积大小')
    plt.ylabel('餐厅利润')
    plt.title('所有数据的分布情况')
    plt.show()
    #plt.clf()
    #画完记得清图，为之后的画做准备，不清不会出问题
    
    #开始训练
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    print('This is the map using training data:\n')
    #散点图，红色表示测试集的点
    plt.scatter(X_test, Y_test, color='red')
    #线图，蓝色表示对测试集进行预测的结果
    plt.plot(X_test, Y_pred, color='blue')
    plt.xlabel('城市面积大小')
    plt.ylabel('餐厅利润')
    plt.title('测试集以及预测结果')
    plt.show()
    #plt.clf()
    
    #预测在面积大小为3.1415的城市开一家餐厅的预计利润
    incomes = regressor.predict([[3.1415]])
    income = incomes[0][0]
    print('面积大小为3.1415的城市开一家餐厅的预计利润为：%.4f\n' %income)