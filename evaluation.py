# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:50:36 2020

@author: zxx
"""

'''
衡量预测效果
groundTruth:真实结果
result:预测向量  此量经过了+testMaskVector操作处理  已经剔除掉训练集中用户有过反映的量
topN：取几个预测结果
'''
def computeTopNAccuracy(groundTruth,result,topN):
#    result = np.squeeze(result)
    result=result.tolist()
    for i in range(len(result)):
        result[i]=(result[i],i)
    result.sort(key=lambda x:x[0],reverse=True)
    #print(result)
    hit=0
    for i in range(topN):
        if(result[i][1] in groundTruth):hit=hit+1
    return hit
#computeTopNAccuracy(0,[1,2,5,9,100,-5,6,0],0)
#computeTopNAccuracy(testSet[testUser], result, recommendAmount)
#(groundTruth,result,topN) = (testSet[testUser], result, recommendAmount)

def computePoi(groundTruth, result, topN, target_item):
    
    result=result.tolist()
    for i in range(len(result)):
        result[i]=(result[i],i)
    result.sort(key=lambda x:x[0],reverse=True)
    #print(result)
    hitR=0
    
    if (target_item in groundTruth):
        hitR=hitR+0
    else:
        for i in range(topN):
#            print(result[i][1])
            if target_item == result[i][1]:
                hitR=hitR+1
    
    
    return hitR


import math
def computeTopNAccuracy2(groundTruth,result,topN):
    result=result.tolist()
    for i in range(len(result)):
        result[i]=(result[i],i)
    result.sort(key=lambda x:x[0],reverse=True)
    #print(result)
    hit=0
    dcg = 0
    idcg = 0
    idcgCount=len(groundTruth)
    for i in range(topN):
        if(result[i][1] in groundTruth):
            hit=hit+1
            dcg+=1/math.log2(i+2)
        if(idcgCount>0):
            idcg += 1/math.log2(i+2)
            idcgCount-=1
    return hit/topN,hit/len(groundTruth),dcg/idcg
#computeTopNAccuracy(0,[1,2,5,9,100,-5,6,0],0)

