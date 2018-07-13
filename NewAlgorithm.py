#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from sklearn import tree
from numpy import *
import math
from sklearn.datasets import load_files
import re
from sklearn.feature_extraction.text import TfidfVectorizer


weakClassArr = []  # 全局变量，用于存储每次训练得到的弱分类器
classifierWeightArr = []  #弱分类器的权重
instanceWeightArr = []  #样本的权重
entropy = [] #熵
dataArr = [] #U集，即全部数据
classLabels = [] #全部数据的标签
lArr = [] #L集
classLabelsOfL = [] #L集的标签


# new algorithm
def newAlgorithmTrain(dataArr, classLabels,numIt=40):
    #T轮训练
    for t in range(numIt):
        #find the weight for h_t
        entropy = computeQx(dataArr) #得到U中每个x的熵

        sampleBasedQx(dataArr,entropy) #simple and label k instances from U

        computeWeight(h_t, )#evaluate h_t using importance sampling...

        #let w_t be the weight of h_t

        #add w_t*h_t to H_{t-1} to obtain H_{t}

        #train the next weak classifier
        #add the sample to L
        #train the next weak classifier h_{t+1} using the enlarged L and go to {*}
        trainNextClf(lArr, classLabelsOfL, t)

#随机抽取样本  dataArr为所有样本数组
def randomSamples(dataArr, labelArr):
    """
    :param dataArr:
    :param labelArr:
    :return:
    """
    #将抽取的样本从U集中删除，添加进L集中
    len = labelArr.len()
    indexList = range(len)
    randomIndex = random.sample(indexList,10)
    for i in enumerate(randomIndex):
        ranDataArr.apppend(dataArr[i])
        ranLabelArr.apppend(labelArr[i])

    # LArr = list(set(dataArr).difference(set(ranDataArr)))

    return ranDataArr, ranLabelArr

#根据q(x)采样和标记样本
def sampleBasedQx(dataArr, entropy, labelArr,  numSample):
    """
    :param dataArr: U集数组
    :param entropy: U集中x的熵
    :param labelArr: 标签
    :param numSample: 采样数量
    :return:
    """
    #simple and label k instances from U
    #将抽取的样本从U集中删除，添加进L集中
    total = sum(entropy)  # 权重求和
    ra = []
    for i in range(numSample):
        ra.append(random.uniform(0, total))  # 在0与权重和之前获取一个随机数

    curr_sum = 0
    ret = None
    keys = dataArr.keys()

    for rw in ra:
        for k in keys:
            curr_sum += entropy[k]  # 在遍历中，累加当前权重值
            if rw <= curr_sum:  # 当随机数<=当前权重和时，返回权重key
                ret.append(k)
                break

    return k_instances

#计算q(x)
def computeQx(dataArr):
    for index, item in enumerate(dataArr):
        prediction = H_{t-1}.predict_proba(item)  #x分类正负类的概率 prediction为[[0., 1.]]
        #计算熵
        entropy[index] = -prediction[0][0]*math.log(prediction[0][0])-prediction[0][1]*math.log(prediction[0][1])
    return entropy

#计算弱分类器权重
def computeWeight(h_t, sampleArr, sampleLabelArr, instanceWeightArr, Qx):
    """
    :param h_t: 当前弱分类器
    :param sampleArr: 当前采样数据集L
    :param sampleLabelArr: 当前采样数据集L的标签
    :param instanceWeightArr: 当前已有的L集合的样本权重
    :param Qx: entropy of H_{t-1}(x)，Qx这里针对于整个enlargedLArr，Qx是个数组
    :return: 弱分类器权重
    """
    err = 0
    m = shape(sampleArr)[0]
    if m != shape(sampleLabelArr[0]) :
        print "Array size mismatch"
    #instanceWeightArr = ones((m, 1)) / m  # 数据集L权重初始化为1/m
    for index, x in enumerate(sampleArr) :
        if sampleLabelArr[index] != h_t.predict(x):
            err = err + instanceWeightArr[index] * Qx[index] #这里要使sampleArr和instanceWeightArr一样大才可以

    weight = math.log((1.0-err)/err)
    return weight #返回该弱分类器的权重

#训练下一个弱分类器
def trainNextClf(lArr, classLabelsOfL, t):
    '''

    :param lArr: enlarged L集
    :param classLabelsOfL: enlarged L集标签
    :param t: 当前轮
    :return: 新的弱分类器h_{t+1}
    '''
    clf = tree.DecisionTreeClassifier()
    h_next = clf.fit(lArr, classLabelsOfL)
    weakClassArr[t+1] = h_next

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
    # feature_names = tf.get_feature_names()
    return X_train_tfidf, train_data.target

if __name__ == '_main_':
    #初始化，随机采样训练h_1

    dataArr, labelArr = preTreatment()  #所有数据的数组
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