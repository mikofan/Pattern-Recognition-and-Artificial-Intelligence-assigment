'''
仅用于多元线性回归的模块

输入：

    文件名

输出：

    1 数据集的样本分布结果（可视化）

    2 线性回归拟合结果（可视化）

    3 预测面积为2000卧室数量为1的房屋的成交价格

'''

#未完成
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def start_MLR(fileName):
    #读取文件，因为本txt没有列名，所以header=None
    dataset = pd.read_csv(fileName, header=None)
    
    #提取特征，0列为面积，1列为利润，X、Y为n行1列的列表，需要用函数转换
    space_feature = dataset[0]
    bedroom_feature = dataset[1]
    price_feature = dataset[2]
    X = np.reshape(space_feature.values, (-1, 1))
    Y = np.reshape(bedroom_feature.values, (-1, 1))
    Z = np.reshape(price_feature.values, (-1, 1))
    
    #画出数据集
    print("This is the map of datasets of space, bedroom and price:\n")
    plt.scatter(X, Z, s=Y, color='red')
    plt.show()
    plt.clf()
    #画完记得清图，为之后的画做准备
    
    #开始训练
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=0.25, random_state=0)
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    print('This is the map using training data:\n')
    #散点图，红色表示测试集的点
    plt.scatter(X_test, Y_test, color='red')
    #线图，蓝色表示对测试集进行预测的结果
    plt.plot(X_test, Y_pred, color='blue')
    plt.show()
    plt.clf()
    
    #预测在面积大小为3.1415的城市开一家餐厅的预计利润
    incomes = regressor.predict([[3.1415]])
    income = incomes[0][0]
    print('面积大小为3.1415的城市开一家餐厅的预计利润为：%.4f\n' %income)