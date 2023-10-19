'''
仅用于多元线性回归的模块

输入：

    文件名

输出：

    1 数据集的样本分布结果（可视化）

    2 线性回归拟合结果（可视化）

    3 预测面积为2000卧室数量为1的房屋的成交价格

'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
#负号正常显示
plt.rcParams['axes.unicode_minus'] = False

def start_MLR(fileName):
    dataset = pd.read_csv(fileName, header=None)
    space_room_feature = dataset[[0, 1]]
    price_feature = dataset[[2]]
    
    X = space_room_feature.values
    Y = price_feature.values
    
    #画出数据集
    plt.scatter(X[:,0], Y)
    plt.xlabel('面积大小')
    plt.ylabel('价格')
    plt.title('面积与房子价格的关系')
    plt.show()
    
    plt.scatter(X[:,1], Y)
    plt.xlabel('卧室数量')
    plt.ylabel('价格')
    plt.title('卧室数量与房子价格的关系')
    plt.show()
    
    #划分训练与测试集，训练模型
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)
    
    Y_pred = regressor.predict(X_test)
    
    #展示训练好后的预测结果
    plt.scatter(X_test[:,0], Y_test, color='red')
    plt.plot(X_test[:,0], Y_pred, color='blue')
    plt.xlabel('面积大小')
    plt.ylabel('价格')
    plt.title('面积与房子价格的关系预测结果')
    plt.show()
    
    plt.scatter(X_test[:,1], Y_test, color='red')
    plt.plot(X_test[:,1], Y_pred, color='blue')
    plt.xlabel('卧室数量')
    plt.ylabel('价格')
    plt.title('卧室数量与房子价格的关系预测结果')
    plt.show()
    
    prices = regressor.predict([[2000, 1]])
    price = prices[0][0]
    print('面积为2000卧室数量为1的房屋的成交价格: %.0f\n' %price)