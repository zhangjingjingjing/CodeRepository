# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection 
from sklearn.neighbors import KNeighborsClassifier

#===========================
#根据文件目录，读取文件数据
#===========================
def load_data(file):               
    pddata = pd.read_csv(file,header=None)  #读取文件
    orig_data=pddata.values #获取文件的数据
    rows=orig_data.shape[0] #获取数据的行数
    cols=orig_data.shape[1] #获取数据的列数
    
    #数据预处理
    for i in range(rows):
        for j in range(cols):
            if(orig_data[i:i+1][0][j]!='?'):
                orig_data[i:i+1][0][j] = int(orig_data[i:i+1][0][j]) #把数据从string类型转换成int类型
            else:
                orig_data[i:i+1][0][j]= np.nan  #用np.nan替代数据中的‘？’，把np.nan作为缺失值处理
                
    #用指定的方法计算数据集中的每个缺失值，然后填充数据          
    impute = preprocessing.Imputer(strategy='most_frequent')
    data = impute.fit_transform(orig_data)
    return data
    

#========================
#处理训练集
#========================
def load_training_set(data):
    digits = list()    #数据
    labels = list()    #标签
    rows=data.shape[0] #获取数据的行数
    for i in range(rows):
        digit = list()
        digit.extend(list(data[i:i+1][0]))  #获取每行数据作为单个数据
        orig_label = digit.pop()            #获取每一个数据的标签
        label = orig_label.astype(np.int32) #将标签转换为int类型
        labels.append(label)         #将单个标签扩展到总标签里
        digits.append(digit)         #将单个数据扩展到数据集里
            
    digits = np.array(digits)         #将数据集列表转换为数组 
    labels = np.array(labels).ravel() #将标签集转换为1维数组
    
    #分成85%的训练集，15%的验证集
    digits_train, digits_verfct, labels_train, labels_verfct = model_selection.train_test_split(digits,labels,test_size=0.15,random_state=5)
    return digits_train, digits_verfct, labels_train, labels_verfct 


#========================
#处理测试集
#========================
def load_test_set(data):  
    digits = list()       #数据
    rows=data.shape[0]    #获取数据的行数
    for i in range(rows):
        digit = list()
        digit.extend(list(data[i:i+1][0]))  #获取每行数据作为单个数据
        digits.append(digit)                #将单个数据扩展到数据集里
    digits = np.array(digits)               #将数据集列表转换为数组 
    return digits


#==============================
# 将测试集的结果保存到csv文件里
#==============================
def csv_save(answer):
    number = np.arange(1,1799)
    save = pd.DataFrame({'id':number,'y':answer})
    save.to_csv('results.csv',index=False)


if __name__ == '__main__':    
    #================加载训练集===========================#
    traning_set = load_data('train.csv')     #加载训练集
    digits_train, digits_verfct, labels_train, labels_verfct = load_training_set(traning_set)  #得到训练集的数据和标签，验证集的数据和标签
    #================训练训练集===========================#
    knn = KNeighborsClassifier(n_neighbors = 54)     #选用KNN模型
    knn.fit(digits_train,labels_train)               #对数据进行训练
    score = knn.score(digits_verfct, labels_verfct)  #计算正确率
    print(score) #输出训练模型的正确率
    #================加载测试集===========================#
    test_set = load_data('test.csv') 
    testDigits=load_test_set(test_set)
    #================预测测试集===========================#
    result = knn.predict(testDigits)
    #================保存预测结果=========================#
    csv_save(result)
       

