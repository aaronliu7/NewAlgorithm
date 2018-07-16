#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from sklearn import tree
import math
import random
from sklearn.datasets import load_files
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from Instance import Instance


# new algorithm
def newAlgorithmTrain(Unlabel_list, Label_list ,weakClassArr, classifierWeightArr, numIt=20):
    '''
    :param Unlabel_list: 全部数据
    :param numIt: 轮数
    :return: H_T
    '''
    #T轮训练
    for t in range(numIt):
        #find the weight for h_t
        Unlabel_list = computeInstanceWeight(Unlabel_list, weakClassArr, classifierWeightArr) #计算样本权重

        numSample = 5
        sample_list = sampleBasedQx(Label_list, Unlabel_list, numSample) #simple and label k instances from U

        classifierWeightArr[t] = computeWeight(weakClassArr[t], sample_list)#evaluate h_t using importance sampling...

        #let w_t be the weight of h_t

        #add w_t*h_t to H_{t-1} to obtain H_{t}

        #train the next weak classifier
        #add the sample to L
        #train the next weak classifier h_{t+1} using the enlarged L and go to {*}
        weakClassArr[t+1] = trainNextClf(Label_list)


#随机抽取样本  dataArr为所有样本数组
def randomSamples(datalist):
    """
    :param datalist: 数据及其标签
    :return:
    """
    l = len(datalist)
    indexList = list(range(l))
    randomIndex = random.sample(indexList, 10)
    sample_ins_list = []
    for i in randomIndex:
        sample_ins_list.append(datalist[i])

    return sample_ins_list


#根据q(x)采样和标记样本
def sampleBasedQx(Label_list, Unlabel_list, numSample):
    """
    :param numSample: 需要采样的个数
    :return: sample_list
    """

    sample_list = []
    total = 0.0
    for ins in Unlabel_list:
        total += ins.weight
    ra = []
    for i in range(numSample):
        ra.append(random.uniform(0, total))  # 在0与权重和之前获取一个随机数

    curr_sum = 0
    keys = Unlabel_list.keys()

    for rw in ra:
        for k in keys:
            curr_sum += Unlabel_list[k].weight  # 在遍历中，累加当前权重值
            if rw <= curr_sum:  # 当随机数<=当前权重和时，返回权重key
                sample_list.append(Unlabel_list[k])
                del Unlabel_list[k]
                Label_list.append(Unlabel_list[k])
                continue

    return sample_list


# 计算样本权重
def computeInstanceWeight(Unlabel_list, weakClassArr, classifierWeightArr):

    weightOfAll = 0 #所有样本的权重，用于归一化
    for index, instance in enumerate(Unlabel_list):
        probaOf0, probaOf1 = generateHt_1x(instance,weakClassArr, classifierWeightArr)  #H_{t-1}(x)
        #计算熵
        instance.weight = -probaOf0*math.log(probaOf0)-probaOf1*math.log(probaOf1)
        weightOfAll += instance.weight
    #归一化
    for instance in Unlabel_list:
        instance.weight = instance.weight/weightOfAll

    return Unlabel_list


#Generate H_{t-1}(x) for each x in U
def generateHt_1x(instance, weakClassArr, classifierWeightArr):
    '''
    :param instance: 样本x
    :param weakClassArr: 弱分类器数组
    :param classifierWeightArr: 弱分类器权重
    :return: probaOf0, probaOf1 H_{t-1}(x)预测为正类和负类的概率
    '''

    probaOf0 = 0
    probaOf1 = 0

    if len(weakClassArr) != len(classifierWeightArr):
        print('!!!!!!!!!!弱分类器个数与其权重个数不对应!!!!!!!!!!!!!')
    else:
        for i in range(len(weakClassArr)):
            prediction = weakClassArr[i].predict_proba(instance) #x分类正负类的概率 prediction为[[0., 1.]]
            probaOf0 += prediction[0][0]*classifierWeightArr[i]
            probaOf1 += prediction[0][1]*classifierWeightArr[i]
        return probaOf0, probaOf1


#计算弱分类器权重
def computeWeight(h_t, samplelist):
    """
    :param h_t: 当前弱分类器
    :param samplelist: 当前采样的数据集
    :return: 弱分类器权重
    """
    err = 0
    for x in samplelist :
        if x.label != h_t.predict(x.data):
            err = err + 1/x.weight

    weight = math.log((1.0-err)/err)
    return weight #返回该弱分类器的权重


#训练下一个弱分类器
def trainNextClf(lArr, classLabelsOfL):
    '''

    :param lArr: enlarged L集
    :param classLabelsOfL: enlarged L集标签
    :param t: 当前轮
    :return: 新的弱分类器h_{t+1}
    '''
    clf = tree.DecisionTreeClassifier()
    h_next = clf.fit(lArr, classLabelsOfL)
    return h_next


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
    return X_train_tfidf.toarray(), train_data.target


#根据instance对象列表分成数据列表和标记列表
def generateTrainArr(instanceArr):
    tempDataArr = []
    tempLabelArr = []
    for i, instance in enumerate(instanceArr):
        tempDataArr.append(instance.data)
        tempLabelArr.append(instance.label)
    return tempDataArr, tempLabelArr


if __name__ == '__main__':

    weakClassArr = []  # 用于存储每次训练得到的弱分类器
    classifierWeightArr = []  # 弱分类器的权重
    Unlabel_list = []  # U集，即全部数据
    Label_list = []  # L集

    """
    预处理 返回的样本数据和标签存储在数据结构Instance中
    每一行的数据格式为稀疏矩阵
    标签是一个列表
    """

    datalist = []
    dataArr, labelArr = preTreatment()  #所有数据 以及标签
    for i in range(len(labelArr)):
        ins = Instance(dataArr[i], labelArr[i])
        datalist.append(ins)

    ini_ins_list = randomSamples(datalist)  #初始随机抽取样本
    Label_list = ini_ins_list
    Unlabel_list =  list(set(datalist).difference(set(Label_list)))

    tempDataArr, tempLabelArr = generateTrainArr(ini_ins_list)  #将instance对象列表生成数据和标签列表
    print(type(tempDataArr[1]))

    clf = tree.DecisionTreeClassifier()
    h_1 = clf.fit(tempDataArr, tempLabelArr)
    weakClassArr.append(h_1)  #初始弱分类器
    classifierWeightArr.append(1.0) #初始弱分类器权重赋初始值

    newAlgorithmTrain(Unlabel_list, Label_list ,weakClassArr, classifierWeightArr)


#哈哈哈



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