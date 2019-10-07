# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 读取和处理数据
def init_data():
    pdData=pd.read_csv('HTRU_2_train.csv',header=None,names=['feature1','feature2','labels'])
    pdData.insert(0, 'Ones', 1)
    orig_data=pdData.values
    cols=orig_data.shape[1] #找出列的数目
    dataMatIn=orig_data[:,0:cols-1] #1-倒数第2列的数据
    classLabels=orig_data[:,cols-1:cols] #倒数第1列的数据
    return dataMatIn, classLabels

# sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

# 梯度下降
def grad_descent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn) #把list特征向量转换为python matrix矩阵（m,n）
    labelMat = np.mat(classLabels) #同样转为matrix矩阵
    m, n = np.shape(dataMatrix) #获取行、列数
    weights = np.ones((n, 1)) #初始化回归系数，计算每个参数前的权重w的值
    alpha = 0.02 #定义步长
    maxCycle = 500 #定义最大循环次数
    
    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights) #sigmoid函数
        weights = weights - alpha * dataMatrix.transpose() * (h-labelMat) #梯度 
    return weights

# 可视化
def plotBestFIt(weights):
    dataMatIn, classLabels = init_data()
    n = np.shape(dataMatIn)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if classLabels[i] == 1:
            xcord1.append(dataMatIn[i][1])
            ycord1.append(dataMatIn[i][2])
        else:
            xcord2.append(dataMatIn[i][1])
            ycord2.append(dataMatIn[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=5, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=5, c='green')
    x = np.arange(0, 200, 0.01) #创建0到200之前，步长为0.01的数据，不包括最后的200
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]  #设置最佳拟合直线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
# 得出测试集的结果
def pulsarTest():
    trainData=pd.read_csv('HTRU_2_test.csv',header=None,names=['feature1','feature2'])
    trainData.insert(0, 'Ones', 1)
    orig_data=trainData.values
    rows=orig_data.shape[0]
    answer = []
    for i in range(rows):
        prob = sigmoid(orig_data[i] * weights)                               
        if prob > 0.5:                                               
            answer.append(1)
        else:
            answer.append(0)
    return answer

# 将测试集的结果保存到csv文件里
def csv_save(answer):
    number = np.arange(1,701)
    save = pd.DataFrame({'id':number,'y':answer})
    save.to_csv('result2.csv',index=False)

if __name__ == '__main__':
    dataMatIn, classLabels = init_data()
    weights = grad_descent(dataMatIn, classLabels)
    plotBestFIt(weights)
    answer = pulsarTest()
    csv_save(answer)


    





