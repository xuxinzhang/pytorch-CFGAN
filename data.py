# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:50:26 2020

@author: zxx
"""


# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) the index should be continuous, without any empy index.
"""

import random
import operator
import numpy as np
import codecs
from collections import defaultdict
from operator import itemgetter
import collections
import torch
from torch.autograd import variable

import pandas as pd



def loadUseInfo(trainFile,splitMark):
    print(trainFile)
    UseInfo = pd.DataFrame(columns=["userId","useAge","useGender","useOccupation","useZipcode"])
    index = 0
    for line in open(trainFile):

        userId, useAge, useGender, useOccupation, useZipcode = line.strip().split(splitMark)
        UseInfo.loc['%d'%index]=[userId, useAge, useGender, useOccupation, useZipcode]
        index = index + 1
        
    UseInfo.to_csv("data/ml-100k/userInfo.csv",index=False)
    print("Demographic information about the users loading done: %d users" % (index))

    return UseInfo 

def loadItemInfo(trainFile,splitMark):
#    trainFile = "data/ml-100k/u.item"
#    splitMark =  "|"

    print(trainFile)
    ItemInfo = pd.DataFrame(columns=["movie_id","movie_title","release_date","video_release_date",
                                     "IMDb_URL","unknowngenres","Action","Adventure",
                                     "Animation","Childrens","Comedy","Crime",
                                     "Documentary","Drama","Fantasy","Film_Noir",
                                     "Horror","Musical","Mystery","Romance",
                                     "Sci_Fi","Thriller","War","Western"])
    index = 0
    for line in open(trainFile,encoding = 'ISO-8859-1'):
#        print(line)
        
        ItemInfo.loc['%d'%index]=line.strip().split(splitMark)
        index = index + 1


    ItemInfo.to_csv("data/ml-100k/itemInfo.csv")
    print("Information about the items loading done: %d users" % (index))

    return ItemInfo  







def splitData():
    random.seed(0)
    print("start")
    fp1 = open("data/ml_small/train.csv", mode="w")
    fp2 = open("data/ml_small/test.csv", mode="w")
    for line in open("data/ml_small/ratings.csv"):
        if (random.randint(0, 8) == 0):
            fp2.writelines(line)
        else:
            fp1.writelines(line)
    print("end")
    fp1.close()
    fp2.close()



def loadData(trainFile,splitMark):
    #trainFile = path + "/" + benchmark + "/" + benchmark + ".train"
    print(trainFile)

    DataSet = defaultdict(list) #字典默认值是一个list    trainSet['key_new'] 是个list
    max_u_id = -1
    max_i_id = -1

    for line in open(trainFile):
        userId, itemId, rating,_ = line.strip().split(splitMark)

        userId = int(userId)
        itemId = int(itemId)

        # note that we regard all the observed ratings as implicit feedback
        DataSet[userId].append(itemId)

        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)

    for u, i_list in DataSet.items():
        i_list.sort()

    userCount = max_u_id + 1
    itemCount = max_i_id + 1

    print(userCount)
    print(itemCount)

    print("Training data loading done: %d users, %d items" % (userCount, itemCount))

    return DataSet, userCount, itemCount  #此处userCount itemCount并不能代表真实值  因为可能小于测试集合中的userCount itemCount






'''加载训练集
trainFile  str 训练集文件名
splitMark  str 文件一行中的分隔符
'''
def loadTrainingData(trainFile,splitMark):
    #trainFile = path + "/" + benchmark + "/" + benchmark + ".train"
    print(trainFile)

    trainSet = defaultdict(list) #字典默认值是一个list    trainSet['key_new'] 是个list
    max_u_id = -1
    max_i_id = -1

    for line in open(trainFile):
        userId, itemId, rating,_ = line.strip().split(splitMark)

        userId = int(userId)
        itemId = int(itemId)

        # note that we regard all the observed ratings as implicit feedback
        trainSet[userId].append(itemId)

        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)

    for u, i_list in trainSet.items():
        i_list.sort()

    userCount = max_u_id + 1
    itemCount = max_i_id + 1

    print(userCount)
    print(itemCount)

    print("Training data loading done: %d users, %d items" % (userCount, itemCount))

    return trainSet, userCount, itemCount  #此处userCount itemCount并不能代表真实值  因为可能小于测试集合中的userCount itemCount

'''
装载测试集数据
testSet [1:[...],2[...], ...]  defaultdict(list)
GroundTruth   [[],[], ...] 只装着item
'''
def loadTestData(testFile,splitMark):
    testSet = defaultdict(list)
    for line in open(testFile):
        userId, itemId, rating,_ = line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)

        # note that we regard all the ratings in the test set as ground truth
        testSet[userId].append(itemId)

    GroundTruth = []
    for u, i_list in testSet.items():
        tmp = []
        for j in i_list:
            tmp.append(j)

        GroundTruth.append(tmp)

    print("Test data loading done")

    return testSet, GroundTruth

''' 返回量trainVector testMaskvector batchCount
testMaskVector 与trainVector相对应  -99999对应1
testMaskVector + 预测结果     （然后取TOP N  相当于去掉了本来就是1的item  拿到了真实有用的预测item）
'''
def to_Vectors(trainSet, userCount, itemCount, userList_test, mode):
    # assume that the default is itemBased

    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount #改动  直接写成userCount
    if mode == "itemBased":#改动  itemCount userCount互换   batchCount是物品数
        userCount = itemCount
        itemCount = batchCount
        batchCount = userCount

    trainDict = defaultdict(lambda: [0] * itemCount)

    for userId, i_list in trainSet.items():
        for itemId in i_list:
            testMaskDict[userId][itemId] = -99999
            if mode == "userBased":
                trainDict[userId][itemId] = 1.0
            else:
                trainDict[itemId][userId] = 1.0

    trainVector = []
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])

    print("Converting to vectors done....")

    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount
def getItemCount(fileName):
    itemCount=0
    for line in open(fileName):
        L=line.split(",")
        itemCount=max(itemCount,int(L[1]))
    return itemCount

#if __name__=="__main__":
#    trainSet, userCount, itemCount=loadTrainingData("data/ml-100k/u1.base","\t")
#    userCount=943+1
#    itemCount=1682+1
#    testSet, GroundTruth=loadTestData("data/ml-100k/u1.test","\t")
#    userList_test = list(testSet.keys())
#    trainVector,testMaskVector,batchCount=to_Vectors(trainSet,userCount,itemCount,userList_test,"userBased")





'''

def loadTrainingData(trainFile,splitMark):
    trainFile = "data/ml-100k/u1.base"
    splitMark = "\t"
#    trainSet, userCount, itemCount = data.loadTrainingData("data/ml-100k/u1.base", "\t")
    print(trainFile)
    
    traindata = pd.DataFrame(columns=["userId","itemId","rating","timestamp"])
    index = 0
    for line in open(trainFile):
#        print(index)
        traindata.loc['%d'%index]=line.strip().split(splitMark)
        index = index + 1
    
    u_count = max(traindata["userId"].values)
    i_count = max(traindata["itemId"])
    traindata.to_csv("data/ml-100k/traindata.csv",index=False)
    print(u_count)
    print(i_count)
    print("Training data loading done: %d users, %d items" % (u_count, i_count))

    return traindata, u_count, i_count




def loadTestData(testFile,splitMark):
    trainFile = "data/ml-100k/u1.test"
    splitMark = "\t"
#    trainSet, userCount, itemCount = data.loadTrainingData("data/ml-100k/u1.base", "\t")
    print(trainFile)
    
    testdata = pd.DataFrame(columns=["userId","itemId","rating","timestamp"])
    index = 0
    for line in open(trainFile):
        print(index)
        testdata.loc['%d'%index]=line.strip().split(splitMark)
        index = index + 1
        
#    u_count = max(testdata["userId"])
#    i_count = max(testdata["itemId"])
    testdata.to_csv("data/ml-100k/testdata.csv",index=False)
#    print(u_count)
#    print(i_count)
    print("Test data loading done" )

    return testdata
'''