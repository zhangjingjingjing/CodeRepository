import os
from PIL import Image
from collections import defaultdict  
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#===============================
# 获取图片中像素点数量最多的像素
#===============================
def get_threshold(image):
    pixel_dict = defaultdict(int)
    
    # 像素及该像素出现次数的字典
    rows, cols = image.size
    for i in range(rows):
        for j in range(cols):
            pixel = image.getpixel((i, j))
            pixel_dict[pixel] += 1

    count_max = max(pixel_dict.values()) # 获取像素出现出多的次数
    pixel_dict_reverse = {v:k for k,v in pixel_dict.items()}
    threshold = pixel_dict_reverse[count_max] # 获取出现次数最多的像素点

    return threshold


#===============================
# 按照阈值进行二值化处理
# threshold: 像素阈值
#===============================
def get_bin_table(threshold):
    # 获取灰度转二值的映射table
    table = []
    for i in range(256):
        rate = 0.1 # 在threshold的适当范围内进行处理
        if threshold*(1-rate)<= i <= threshold*(1+rate):
            table.append(1)
        else:
            table.append(0)
    return table


#=================================
# 去掉二值化处理后的图片中的噪声点
#=================================
def cut_noise(image):
    rows, cols = image.size # 图片的宽度和高度
    change_pos = [] # 记录噪声点位置

    # 遍历图片中的每个点，除掉边缘
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # pixel_set用来记录该点附近的黑色像素的数量
            pixel_set = []
            # 取该点的邻域为以该点为中心的九宫格
            for m in range(i-1, i+2):
                for n in range(j-1, j+2):
                    if image.getpixel((m, n)) != 1: # 1为白色,0位黑色
                        pixel_set.append(image.getpixel((m, n)))

            # 如果该位置的九宫内的黑色数量小于等于4，则判断为噪声
            if len(pixel_set) <= 4:
                change_pos.append((i,j))

    # 对相应位置进行像素修改，将噪声处的像素置为1（白色）
    for pos in change_pos:
        image.putpixel(pos, 1)

    return image # 返回修改后的图片


#============================================
# 传入参数为图片路径，返回结果为处理后的图片
#============================================
def picture_processing(img_path):
    image = Image.open(img_path) # 打开图片文件
    imgry = image.convert('L')  # 转化为灰度图
#    plt.subplot(311)
#    plt.imshow(imgry)
    # 获取图片中的出现次数最多的像素，即为该图片的背景
    max_pixel = get_threshold(imgry)

    # 将图片进行二值化处理
    table = get_bin_table(threshold=max_pixel)
    out = imgry.point(table, '1')
#    plt.subplot(312)
#    plt.imshow(out)
    
    # 去掉图片中的噪声（孤立点）
    out = cut_noise(out)
#    plt.subplot(313)
#    plt.imshow(out)
    
    return out


#============================================
#图片字符切割 
#============================================
def get_crop_imgs(img):
    child_img_list = []
    for i in range(5):
        x = 0+i*30  
        y = 0
        child_img = img.crop((x, y, x + 30, y + 30))
        child_img_list.append(child_img)
    return child_img_list


#============================================
#创建文件夹，以便里面存放图像的数字文件 
#============================================
def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    # 判断结果，如果不存在则创建目录
    # 创建目录操作函数
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False
 
    
#===================================================
#统计文件夹里面关于各个字符的文件个数，以便数字文件排序
#===================================================
def num_name(ndir,nei):
    num = 0
    transfername = nei
    name_dict = defaultdict(int)
    for file in os.listdir(ndir):
        na = file.split('_')[0]  # 图片名称，即图片中的正确文字              
        name_dict[na] += 1
    for i in name_dict.keys():
        if i == nei:
            num = name_dict[nei]
    if num == 0:
        transfer = nei.swapcase()
        #是字母
        if transfer != nei:
            for m in name_dict.keys():
                if m == transfer:
                    transfername = nei+'1'
            for n in name_dict.keys():
                if n == nei+'1':
                    num = name_dict[n]
    return num,transfername


#============================================
#在指定的文件夹中生成各个字符的数字文件（训练集）
#============================================
def digital_file(dirct,file):  
    #对传入的图片进行预处理及切割
    out = picture_processing(file)
    child_img_list = get_crop_imgs(out)
    #文件名字
    file_name = file.split('.')[0]
    #图片名字
    picture_name = file_name[-5:]
    #创建存放图片名字与生成的5个对应数字文件名字的映射关系的字典
    files_name_dict = defaultdict(int) 
    name_list = []
    for i in range(5):
        #进入目录
        os.chdir(dirct)
        #单个字符的名字
        name = picture_name[i]
        num,transfername = num_name(dirct,name)
        #单个字符的数字文件的名字
        digits_file_name = transfername+'_'+str(num)+'.txt'
        name_list.append(digits_file_name)
        #生成单个字符的数字文件
        fileObject = open(digits_file_name, 'w+') 
        #填充数字文件，把图像像素转换为数字
        for m in range(child_img_list[i].size[1]):
            for n in range(child_img_list[i].size[0]):
                pos=(n,m)
                col=child_img_list[i].getpixel(pos)
                if col==0:
                    col=1
                else:
                    col=0
                fileObject.write(str(col))               
            fileObject.write('\n')        
        fileObject.close() 
    #将映射关系存入字典    
    files_name_dict[picture_name] = name_list
    return files_name_dict


#============================================
#在指定的文件夹中生成各个字符的数字文件（测试集）
#============================================
def digital_file2(dirct,file):  
     #对传入的图片进行预处理及切割
    out = picture_processing(file)
    child_img_list = get_crop_imgs(out)
    #文件名字
    file_name = file.split('.')[0]
    #图片名字
    picture_name = file_name[26:]
    #创建存放图片名字与生成的5个对应数字文件名字的映射关系的字典
    files_name_dict = defaultdict(int) 
    name_list = []
    for i in range(5):
        #进入目录
        os.chdir(dirct)
        #单个字符的数字文件的名字
        digits_file_name = picture_name+'_'+str(i)+'.txt'
        name_list.append(digits_file_name)
        #生成单个字符的数字文件
        fileObject = open(digits_file_name, 'w+') 
        #填充数字文件，把图像像素转换为数字
        for m in range(child_img_list[i].size[1]):
            for n in range(child_img_list[i].size[0]):
                pos=(n,m)
                col=child_img_list[i].getpixel(pos)
                if col==0:
                    col=1
                else:
                    col=0
                fileObject.write(str(col))
            fileObject.write('\n')
        fileObject.close() 
    #将映射关系存入字典  
    files_name_dict[picture_name] = name_list
    return files_name_dict


#============================================
# 识别指定文件目录下的图片（训练集）
#============================================
def identify(directory,dirct):
    #创建存放全部图片名字和与之对应生成的5个对应数字文件名字的映射关系的字典
    files_dict = defaultdict(int) 
    # 遍历文件目录下的文件
    for file in os.listdir(directory):
        image_path = '%s/%s'%(directory,file) # 图片路径
        files_name_dict = digital_file(dirct,image_path) 
        
        for k,v in files_name_dict.items():          
            files_dict[k] = v
    return files_dict


#============================================
# 识别指定文件目录下的图片（测试集）
#============================================
def identify2(directory,dirct):
    #创建存放全部图片名字和与之对应生成的5个对应数字文件名字的映射关系的字典
    files_dict = defaultdict(int) 
    # 遍历文件目录下的文件
    for file in os.listdir(directory):
        image_path = '%s/%s'%(directory,file) # 图片路径

        files_name_dict = digital_file2(dirct,image_path) 
        for k,v in files_name_dict.items():          
            files_dict[k] = v
    
    return files_dict
    

#============================================  
#给出文件目录，读取数据，生成训练集
#============================================
def loadData(dir):    
    digits = list() #数据集（数据）
    labels = list() #标签
    if os.path.exists(dir): #判断目录是否存在
        files = os.listdir(dir) #获取目录下的所有文件名
        for file in files:  #遍历所有文件
            label = file.split('_')[0]
            if len(label) == 2:
                label = label[0]
            labels.append(label)   #按照文件名规则，文件名第一位是标签
            with open(dir + '\\' + file) as f:  #通过“目录+文件名”，获取文件内容
                digit = list()
                for line in f:  #遍历文件每一行
                    digit.extend(map(int, list(line.replace('\n', ''))))    #遍历每行时，把数字通过extend的方法扩展
                digits.append(digit)    #将数据扩展进去
    #建立0-z到0-61的映射关系
    label_dict = defaultdict(int)    
    j = 0
    for i in range(len(labels)):
        if labels[i] not in label_dict.keys():                
            label_dict[labels[i]] = j
            j += 1
    #标签转换成0-61
    new_labels = labels.copy()
    for i in range(len(new_labels)):
        for key in label_dict.keys():
            if new_labels[i] ==  key:
                new_labels[i] = label_dict[key]

    digits = np.array(digits)   #数据集
    new_labels = np.array(new_labels).ravel()  #将标签重构成(N, 1)的大小
    return digits, new_labels, label_dict


#============================================  
#对验证集进行验证
#============================================
def predict_verification(files_dict,mktestdata,label_dict):
    #所有图片相对应的数字文件名字的列表
    file_name = files_dict.values()
    
    accuracy = 0
    N = len(files_dict)
    
    #每个图片的最后预测结果
    result = list()
    for single_name in file_name:
        files = list() 
        digits = list()
        #获取5个数字文件的名字
        for i in range(5):
            file = mktestdata+single_name[i]
            files.append(file)
            
        #加载5个数字文件，生成测试集
        for file in files:  #遍历所有文件
            with open(file) as f:  #通过“目录+文件名”，获取文件内容
                digit = list()
                for line in f:  #遍历文件每一行
                    digit.extend(map(int, list(line.replace('\n', ''))))    #遍历每行时，把数字通过extend的方法扩展
                digits.append(digit)    #将数据扩展进去
                
        predicts = ''
        #对验证集进行预测
        for digit in digits:        
            new_digit = []
            new_digit.append(digit)
            pred = knn.predict(new_digit)
            for k,v in label_dict.items():
                pre = ''
                if v==pred:
                    pre = k 
                predicts += pre
        result.append(predicts)

        for k,v in files_dict.items():
            if v==single_name:
                picture_name = k 
                
        print("predict:%s, actual:%s"% (predicts, picture_name))
        if (predicts == picture_name):
            accuracy += 1
            
    print("accuracy:%.5f" %(accuracy / N ))

   
#============================================  
#对测试集进行预测
#============================================
def predict_test(files_dict,mktestdata,label_dict):
    #所有图片相对应的数字文件名字的列表
    file_name = files_dict.values()

    #每个图片的最后预测结果
    result = list()
    sorted_files_dict = defaultdict(int)     
    for single_name in file_name:
        files = list() 
        digits = list()
        #获取5个数字文件的名字
        for i in range(5):
            file = single_name[i]
            files.append(file)
            
        #加载5个数字文件，生成测试集
        for file in files:  #遍历所有文件
            with open(file) as f:  #通过“目录+文件名”，获取文件内容
                digit = list()
                for line in f:  #遍历文件每一行
                    digit.extend(map(int, list(line.replace('\n', ''))))    #遍历每行时，把数字通过extend的方法扩展
                digits.append(digit)    #将数据扩展进去
                        
        predicts = ''
        #对测试集进行预测
        for digit in digits:        
            new_digit = []
            new_digit.append(digit)
            pred = knn.predict(new_digit)
            
            for k,v in label_dict.items():
                pre = ''
                if v==pred:
                    pre = k 
                predicts += pre
        result.append(predicts)
        
        for k,v in files_dict.items():
            if v==single_name:
                picture_name = k 
                
        print("predict:%s, number:%s"% (predicts, picture_name))
        sorted_files_dict[int(picture_name)] = predicts
        
    return sorted_files_dict

    
#==============================
# 将测试集的结果保存到csv文件里
#==============================
def csv_save(answer):
    number = np.arange(0,20000)
    save = pd.DataFrame({'id':number,'y':answer})
    save.to_csv('resultlast3.csv',index=False)
      
    
#===============================================================================
if __name__ == '__main__':    
#    #训练集存放目录
#    traindir = 'E:\\dasanshang\\机器学习框架\\train'
#    #训练集图片的数字文件的存放目录
#    mktraindata="E:\\dasanshang\\机器学习框架\\traindata19994\\"  
#    #创建目录
#    mkdir(mktraindata)
#    #加载训练集图片，生成图片的数字文件训练集
#    identify(traindir,mktraindata)
    
    #处理训练集，得到训练集数据，训练集标签，0-61与大小写字母和10个数字的映射关系
    trainDigits, trainLabels, label_dict = loadData('traindata19994')
    knn = KNeighborsClassifier(n_neighbors = 1)     #选用KNN模型
    knn.fit(trainDigits,trainLabels) 
    
    #测试集存放目录
    testdir = 'E:\\dasanshang\\机器学习框架\\test'
    #测试集图片的数字文件的存放目录
    mktestdata="E:\\dasanshang\\机器学习框架\\testdata20000\\"  
    #创建目录
    mkdir(mktestdata)   
    
    #获取图片名字与生成的对应数字文件名字的映射关系   
    files_dict = identify2(testdir,mktestdata)
    
    #得到预测结果
    sorted_files_dict = predict_test(files_dict,mktestdata,label_dict)

    #对预测结果按照序号排序，得到正确的结果顺序
    sorted_result = list()
    for i in sorted (sorted_files_dict) : 
        sorted_result.append(sorted_files_dict[i])
    
    #保存预测结果到csv文件
    csv_save(sorted_result)
 
    
#==========================================================
# 从训练集中抽出1333张图片作为训练集，抽出667张图片作为验证集，
# 对训练集进行训练，然后预测验证集，得出正确率
#==========================================================
      
#    #包含1333张图片的训练集存放目录
##    traindir = 'E:\\dasanshang\\机器学习框架\\selftrain1333'
#    #包含1333张图片的训练集图片的数字文件的存放目录
##    mktraindata="E:\\dasanshang\\机器学习框架\\traindata1333\\"  
#    #创建目录
##    mkdir(mktraindata)
#    #加载训练集图片，生成训练集图片的数字文件
##    identify(traindir,mktraindata)
    
#    #处理训练集，得到训练集数据，训练集标签，0-61与大小写字母和10个数字的映射关系
#    trainDigits, trainLabels, label_dict = loadData('traindata1333')
#    knn = KNeighborsClassifier(n_neighbors = 1)     #选用KNN模型
#    knn.fit(trainDigits,trainLabels) 
#
#    #从训练集中抽出667张图片作为验证集
#    #验证集的存放目录
#    testdir = 'E:\\dasanshang\\机器学习框架\\selfverify667'
#    #验证集图片的数字文件的存放目录
#    mktestdata="E:\\dasanshang\\机器学习框架\\verifydata667\\"  
#    #创建目录
#    mkdir(mktestdata)
#    
#    #获取图片名字与生成的对应数字文件名字的映射关系
#    files_dict = identify(testdir,mktestdata)
#    
#    #进行预测
#    predict_verification(files_dict,mktestdata,label_dict)
    
    
    
    