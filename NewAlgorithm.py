#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from sklearn import tree
from numpy import *
import math
from sklearn.datasets import load_files
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer


weakClassArr = []  # 全局变量，用于存储每次训练得到的弱分类器以及其输出结果的权重
entropy = [] #熵
#自适应数据加载函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):  #最后一项为label
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# new algorithm
def newAlgorithmTrain(dataArr, classLabels,numIt=40):
    #T轮训练
    for t in range(numIt):
        #find the weight for h_t
        entropy = computeQx(dataArr) #得到U中每个x的熵

        sampleBasedQx(dataArr,entropy) #simple and label k instances from U

        #evaluate h_t using importance sampling...

        #let w_t be the weight of h_t

        #add w_t*h_t to H_{t-1} to obtain H_{t}

        #train the next weak classifier
        #add the sample to L
        #train the next weak classifier h_{t+1} using the enlarged L and go to {*}

#随机抽取样本  dataArr为所有样本数组
def randomSamples(dataArr, labelArr):
    return ranDataArr, ranLabelArr

#根据q(x)采样和标记样本
def sampleBasedQx(dataArr, entropy):
    #simple and label k instances from U
    return k instances


#计算q(x)
def computeQx(dataArr):
    for index, item in enumerate(dataArr):
        prediction = H_{t-1}.predict_proba(item)  #x分类正负类的概率 prediction为[[0., 1.]]
        #计算熵
        entropy[index] = -prediction[0][0]*math.log(prediction[0][0])-prediction[0][1]*math.log(prediction[0][1])
    return entropy

#数据预处理
def preTreatment():
    # 选取参与分析的文本类别
    categories = ['baseball', 'hockey']
    # 从硬盘获取原始数据
    train_data = load_files("data/baseball-hockey",
                            categories=categories,
                            load_content=True,
                            encoding="latin1",
                            decode_error="strict",
                            shuffle=True, random_state=42)

    data = []
    for s in train_data.data:
        o = re.sub(r'([\d]+)', '', s)
        o = o.replace('_', '')
        data.append(o)

    # 使用tf-idf方法提取文本特征
    tf = TfidfVectorizer(min_df=1, analyzer='word', stop_words='english', strip_accents='ascii')
    X_train_tfidf = tf.fit_transform(data)
    feature_names = tf.get_feature_names()
    # 打印特征矩阵规格
    print(type(X_train_tfidf))
    print(X_train_tfidf.shape)
    print(feature_names)
    print(X_train_tfidf[0])
    return array



if _name_ == '_main_':
    #初始化，随机采样训练h_1
    dataArr  #所有数据的数组
    ranDataArr, ranLabelArr = randomSamples(dataArr, labelArr)  #随机抽取样本
    clf = tree.DecisionTreeClassifier()
    h_1 = clf.fit(ranDataArr, ranLabelArr)
    weakClassArr[0] = h_1






# #训练决策树
# dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(dataArr, labelArr)
# #测试
# testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
# prediction = clf.predict_proba(testArr)
# print(prediction)
# print(mat(prediction).T)
# print(mat(testLabelArr).T)
#错误率
# print(1.0*sum(mat(prediction).T != mat(testLabelArr).T)/len(prediction))
# print(type(clf))