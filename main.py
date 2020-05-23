# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:36:44 2020

@author: zxx
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import cfgan
import random
import evaluation
import re
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

import data


from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")




def paint(x,y):
    plt.title("precision")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x, y, "k-o")
    plt.ylim([0, 0.5])
    plt.show()
'''def adjustMaskVector(maskVector,m):
    itemCount=len(maskVector[0])
    for i in range(len(maskVector)):
        for j in range( m ):
            index=random.randint(0,itemCount-1)
            maskVector[i][index]=1'''

def paint2(x,y1,y2,y3):
    plt.title("CFGAN")
    plt.xlabel('epoch')
    #plt.ylabel('')
    plt.plot(x, y2, "k-o", color='red', label='recall', markersize='0')
    plt.plot(x, y1, "k-o",color='black',label='precision',markersize =  '0' )#List,List,List

    plt.plot(x, y3, "k-o",color='green',label='ndcg',markersize =  '0')
    plt.ylim([0, 0.6])
    plt.legend()  # 图例
    plt.rcParams['lines.linewidth'] = 1
    plt.show()

def main(trainSet,userCount,itemCount,testSet,trainVector,testMaskVector,batchCount,target_item,UseInfo_pre,epochCount,pro_ZR,pro_PM,alpha):
#    (epochCount,pro_ZR,pro_PM,alpha) = (1000,300,300,0.5)

#    epochCount,pro_zp = 4000, 1000
#    (epochCount,pro_zp,alpha)=(500,300,0.5)
#    UseInfo_pre = UseInfoPreprocessing(UseInfo)
    
    
    info_shape = UseInfo_pre.shape[1]
#    UseInfo_pre['userId'] = [int(x) for x in UseInfo_pre['userId']]
    UseInfo_pre = UseInfo_pre.values
    
    UseInfo_pre = np.insert(UseInfo_pre,0,[0,0,0,0,0],axis=0)
    UseInfo_pre = torch.tensor(UseInfo_pre.astype(np.float32))
    
    
    X=[] #画图数据的保存
    X2=[]
    precisionList=[]
    precisionList2=[]
    recallList=[]
    ndcgList=[]
    G=cfgan.generator(itemCount, info_shape)
    D=cfgan.discriminator(itemCount, info_shape)

    criterion1 = nn.BCELoss()  # 二分类的交叉熵
    criterion2 = nn.MSELoss()
#    criterion2 = nn.MSELoss(size_average=False)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

    G_step=5
    D_step=2
    batchSize_G = 32
    batchSize_D = 32
    realLabel_G = Variable(torch.ones(batchSize_G))
#    fakeLabel_G = Variable(torch.zeros(batchSize_G))
    realLabel_D = Variable(torch.ones(batchSize_D))
    fakeLabel_D = Variable(torch.zeros(batchSize_D))
    ZR = []
    PM = []
    
    label = Variable(torch.zeros(userCount,2))
    for i in range (userCount):
        if int(trainVector[i,target_item-1]) == 1:
            label[i,1] = 1
    
    
    for epoch in range(epochCount): #训练epochCount次

        if(epoch%100==0):
            ZR = []
            PM = []
            for i in range(userCount):
                ZR.append([])
                PM.append([])
                ZR[i].append(np.random.choice(itemCount,pro_ZR,replace=False))
                PM[i].append(np.random.choice(itemCount,pro_PM,replace=False))
        
            
        for step in range(G_step):#训练G0
            #调整maskVector2\3
#            data1=copy.deepcopy(data)
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            realData = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))

            maskVector2 = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            maskVector3 = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex + batchSize_G]))
            
            useInfo_batch = Variable(copy.deepcopy(UseInfo_pre[leftIndex:leftIndex + batchSize_G]))
            for i in range(len(maskVector2)):
                maskVector2[i][PM[i+leftIndex]] = 1
                maskVector3[i][ZR[i+leftIndex]] = 1
            
            noise_batch = torch.tensor((5 * np.random.random_sample([batchSize_G,100])).astype(np.float32))
#            fakeData=G(noise_batch,label_batch,useInfo_batch)  #CGAN
            fakeData=G(realData,useInfo_batch) 
#            fakeData=G(realData)
            fakeData=fakeData*maskVector2    #要不要考虑目标项
#            g_fakeData_result=D(fakeData,realData,useInfo_batch,label_batch)  #CGAN
            g_fakeData_result=D(fakeData,realData,useInfo_batch)  #CGAN
            
            #RecQ
            g_loss = np.mean(np.log(1.-g_fakeData_result.detach().numpy()+10e-5))+alpha*criterion2(fakeData,maskVector3)
#            print(criterion2(fakeData,maskVector3))
            

            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
        for step in range(D_step):#训练D
            #maskVector1 是PM方法的体现  这里要进行优化  减少程序消耗的内存

            leftIndex=random.randint(1,userCount-batchSize_D-1)
            realData=Variable(copy.deepcopy(trainVector[leftIndex:leftIndex+batchSize_D])) #MD个数据成为待训练数据

            maskVector1 = Variable(copy.deepcopy(trainVector[leftIndex:leftIndex+batchSize_D]))
            
            useInfo_batch = Variable(copy.deepcopy(UseInfo_pre[leftIndex:leftIndex + batchSize_G]))

            for i in range(len(maskVector1)):
                maskVector1[i][PM[leftIndex+i]]=1
#            Condition=realData#把用户反馈数据作为他的特征 后期还要加入用户年龄、性别等信息
#            realData_result=D(realData,useInfo_batch,label_batch) #CGAN
            realData_result=D(realData,realData,useInfo_batch) #CGAN
            
            d_loss_real=criterion1(realData_result,realLabel_D)
            
            noise_batch = torch.tensor((5 * np.random.random_sample([batchSize_G,100])).astype(np.float32))

            fakeData=G(realData,useInfo_batch) #CGAN

            fakeData=fakeData*maskVector1
            fakeData_result=D(fakeData,realData,useInfo_batch) #CGAN

            d_loss_fake=criterion1(fakeData_result,fakeLabel_D)
            
            #原程序
            d_loss_1=d_loss_real+d_loss_fake  
            #RecQ
            d_loss = -np.mean(np.log(realData_result.detach().numpy()+10e-5) + 
                              np.log(1. - fakeData_result.detach().numpy()+10e-5)) + 1e-5*criterion2(fakeData,maskVector3)  #RecQ
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
        if( epoch%10==0):

            hit=0
            hitR = 0
            peopleAmount=len(testSet)
            recommendAmount=5

            
            index=0
            precisions=0
            recalls=0
            ndcgs=0
#            i = 0
            for testUser in testSet.keys():
#                i = i+1
#                print(i)
                data = Variable(copy.deepcopy(trainVector[testUser]))
                label_test = Variable(torch.zeros(1,2))
                label_test[0,1] = data[target_item-1]
                
                noise_index = torch.tensor((5 * np.random.random_sample([1,100])).astype(np.float32))

                useInfo_index = Variable(copy.deepcopy(torch.tensor(np.expand_dims(UseInfo_pre[index], axis=0))))
#                torch.tensor(UseInfo_pre.astype(np.float32))
                result = G(data.reshape(1,1683),useInfo_index) + Variable(copy.deepcopy(testMaskVector[index]))
#                result = G(data.reshape(1,1683)) + Variable(copy.deepcopy(testMaskVector[index]))
                result = result.reshape(1683)
                index+=1
                hit = hit + evaluation.computeTopNAccuracy(testSet[testUser], result, recommendAmount)
                hitR = hitR + evaluation.computePoi(testSet[testUser], result, recommendAmount, target_item)
#                eva = evaluation.computeTopNAccuracy(testSet[testUser], result, recommendAmount)
#                print(evaluation.computeTopNAccuracy(testSet[testUser], result, recommendAmount))
#                if eva != 0:
#                    break
                precision,recall,ndcg=evaluation.computeTopNAccuracy2(testSet[testUser], result, recommendAmount)
                precisions+=precision
                recalls+=recall
                ndcgs+=ndcg
                
            Poi_hit = hitR/peopleAmount
#            precision=hit/(peopleAmount)
            precision=hit/(peopleAmount*recommendAmount)
            precisionList.append(precision)
            X.append(epoch)
#            print('Epoch[{}/{}],d_loss:{:.6f},{:.6f},g_loss:{:.6f},{:.6f},precision:{},poison HitRatio:{}'.format(epoch, epochCount,
#            d_loss_1.item(),d_loss.item(),
#            g_loss_1.item(),g_loss.item(),
#            hit/(peopleAmount*recommendAmount),
#            Poi_hit))
            #hit/(peopleAmount*recommendAmount)
#            plt.figure()
#            paint(X,precisionList)
            
            
            precisions /= peopleAmount
            recalls /= peopleAmount
            ndcgs /= peopleAmount

            precisionList2.append(precisions)
            recallList.append(recalls)
            ndcgList.append(ndcgs)

            X2.append(epoch)
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},precision:{},recall:{},ndcg:{}'.format(epoch, epochCount,
            d_loss.item(),
            g_loss.item(),
            precisions,recalls,ndcgs))
#            plt.figure()
            paint2(X,precisionList2,recallList,ndcgList)

            
    return precisionList



def UseInfoPreprocessing(UseInfo):
#    UseInfo.head()
#    UseInfo.info()
#    UseInfo['useGender'].value_counts().plot.pie(labeldistance = 1.1,autopct = '%1.2f%%',
#                                               shadow = False,startangle = 90,pctdistance = 0.6)
#    
    useGender_dummies = pd.get_dummies(UseInfo['useGender'])
    UseInfo = UseInfo.join(useGender_dummies)
    UseInfo.drop(['useGender'], axis=1, inplace=True)


    # create feature for the alphabetical part of the cabin number
    UseInfo['useOccupationLetter'] = UseInfo['useOccupation'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    # convert the distinct cabin letters with incremental integer values
    UseInfo['useOccupationLetter'] = pd.factorize(UseInfo['useOccupationLetter'])[0]
    UseInfo[['useOccupation','useOccupationLetter']].head()
    UseInfo.drop(['useOccupation'], axis=1, inplace=True)
    
    
    UseInfo['useZipcodeLetter'] = UseInfo['useZipcode'].str.split().str[0]
    UseInfo['useZipcodeLetter'] = UseInfo['useZipcodeLetter'].apply(lambda x: "99999" if not(x.isnumeric()) else x)
    UseInfo['useZipcodeLetter'] = pd.factorize(UseInfo['useZipcodeLetter'])[0]
    UseInfo[['useZipcode','useZipcodeLetter']].head()
    UseInfo.drop(['useZipcode'], axis=1, inplace=True)

    
    # StandardScaler will subtract the mean from each value then scale to the unit varience
    scaler = preprocessing.StandardScaler()
    UseInfo['useAge_scaled'] = scaler.fit_transform(UseInfo['useAge'].values.reshape(-1,1))
    UseInfo.drop(['useAge'], axis=1, inplace=True)
    #可以考虑将年龄binning,然后dummies化
    
    return UseInfo

#import train
if __name__ == '__main__':
    
    dataSet, userCount, itemCount = data.loadData("data/ml-100k/u.data", "\t")
    
    allindex = random.sample(range(1, userCount), userCount-1)
    testindex = random.sample(range(1, userCount), int(0.2*userCount))
    trainindex = np.delete(allindex, testindex)
    
#    trainSet = defaultdict(list)
#    for i in trainindex:
#        trainSet[i] = dataSet[i]
#        
#    testSet = defaultdict(list)
#    for i in testindex:
#        testSet[i] = dataSet[i]

#    traindata = pd.read_csv("data/ml-100k/traindata.csv")
    trainSet, userCount, itemCount = data.loadTrainingData("data/ml-100k/u1.base", "\t")
    userCount = 943 + 1
    itemCount = 1682 + 1
    target_item = 2
#    testdata = pd.read_csv("data/ml-100k/testdata.csv")
#    testdata = data.loadTestData("data/ml-100k/u1.test", "\t")
    testSet, GroundTruth = data.loadTestData("data/ml-100k/u1.test", "\t")
    
    UseInfo = data.loadUseInfo("data/ml-100k/u.user"  , "|")
#    ItemInfo = data.loadItemInfo("data/ml-100k/u.item"  , "|")
    UseInfo_pre = UseInfoPreprocessing(UseInfo)
    UseInfo_pre.drop(['userId'], axis=1, inplace=True)
    userList_test = list(testSet.keys())
    trainVector, testMaskVector, batchCount = data.to_Vectors(trainSet, userCount, itemCount, userList_test, "userBased")
#    plt.figure()
    main(trainSet,userCount,itemCount,testSet,trainVector,testMaskVector,batchCount,target_item,UseInfo_pre,1000,300,300,0.5)


#    trainSet, userCount, itemCount = data.loadTrainingData("data/ml-100k/u2.base", "\t")
#    testSet, GroundTruth = data.loadTestData("data/ml-100k/u2.test", "\t")
#    trainVector, testMaskVector, batchCount = data.to_Vectors(trainSet, userCount, itemCount, userList_test, "userBased")
#    train.main(trainSet,userCount,itemCount,testSet,GroundTruth,trainVector,testMaskVector,batchCount,1000,0.5,0.7,0.03)

    
#7
#300 300 0.1
#8
#300 300 0.05
#    
#    epochCount = 1000
#    pro_zps = [1000,300,100,30,10]
#    alphas = [0.5,0.2,0.1,0.05,0.01]
#    test = 0
#    for i in range(5):
#        for j in range(5):
#            print(test)
#            test = test + 1
#            print(pro_zps[i],pro_zps[i],alphas[j])
#            plt.figure(i*5 + j)
#            main(trainSet,userCount,itemCount,testSet,trainVector,testMaskVector,batchCount,target_item,UseInfo_pre,epochCount,pro_zps[i],pro_zps[i],alphas[j])

            
    
    
'''
trainSet : 943个用户，每一个用户的购买的项目向量
userCount : 944
itemCount : 1683
testSet : 459个用户，每一个用户的购买的项目向量，与trainset不重叠
GroundTruth: 459个用户，每一个用户的购买的项目向量，与testSet内容一致，序号有差异
trainVector: 944*1683的矩阵,第一行第一列全0，第二行开始与trainSet对应
testMaskVector: testMaskVector 与trainVector相对应  -99999对应1
                testMaskVector + 预测结果     （然后取TOP N  相当于去掉了本来就是1的item  拿到了真实有用的预测item）
batchCount： 944

sample N-ZR:
ZR : 944*1*1024,每一个user对应的
PM:

realData : 32*1683
    

这个程序是个没有完成的程序，还需要修改，可以借鉴思路

改进方向：
1：G网络和D网络的输入并不是按照论文来的
2：G的损失计算时权重可改{0.5, 0.25, 0.1, 0.05, 0.01}.
3：{10, 30, 50, 70, 90} ZR PM
4:损失函数的计算是否合理

现在的问题是该输入后和之前应该是没啥出入，我们要考虑如何将结果往我们想要的结果上靠

生成一些含有目标项的投进去，然后算前后的击中率？
这个需要先完成其它攻击方法的复现，相当于我们完成了一个生成数据的过程。

假如我们考虑这个东西作为生成中毒数据的一种手段，那么我们就需要从生成的结果中选出
最好的几次生成结果作为生成的中毒数据，


'''    
    
    
    
    
    
    
    
    
    
    