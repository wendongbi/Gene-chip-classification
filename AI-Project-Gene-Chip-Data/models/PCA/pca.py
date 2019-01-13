#!/usr/env/bin
# -*-coding:utf-8*-
# Copyright: Copyright (c) 2018

# All rights reserved

# Created on 2018-12-18  

# Author:Wendong Bi

# Filename: pca.py

# discription: pca dimention reduction functions.
import numpy as np

# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)     # 按列求均值，即求各个特征的均值
    newData = dataMat-meanVal
    return newData, meanVal

# 去除不重要的维度，num即降维后的维度，percentage是权重
def percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)   # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum*percentage:
            return num

# 求出原始数据集所有的特征值和对应的特征向量，然后从里面选n个最大的特征值对应的特征向量，作为PCA主成分。
def pca(dataMat, percentage=0, k=453):
    dataMat, meanVal = zeroMean(dataMat)
    print("datamat type :" + str(type(dataMat))+str(dataMat.shape))

    print("Now computing covariance matrix...")
    covMat = np.cov(dataMat, rowvar=0)    # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    print("covmat type :" + str(type(covMat))+str(covMat.shape))

    print("Finished. Now solve eigen values and vectors...")
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))# 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    print("eigVals type :" + str(type(eigVals))+str(eigVals.shape))
    print("eigVects type :" + str(type(eigVects))+str(eigVects.shape))

    print("Finished. Now select eigen vectors...")
    if percentage is not 0:
        n = percentage2n(eigVals, percentage)                 # 要达到percent的方差百分比，需要前n个特征向量
    else:
        n = k
    eigValIndice = np.argsort(eigVals)            # 对特征值从小到大排序
    print("eigValIndice type :" + str(type(eigValIndice))+str(eigValIndice.shape))

    n_eigValIndice = eigValIndice[-1:-(n+1):-1]   # 最大的n个特征值的下标，index=-i表示倒数第i个index=i表示正数第i-1个，正数从零开始，倒数从-1开始。
    n_eigVect = eigVects[:, n_eigValIndice]        # 最大的n个特征值对应的特征向量
    print("Finished. Now generating new data...")
    print("n_eigVect type :" + str(type(n_eigVect))+str(n_eigVect.shape))

    lowDDataMat = dataMat*n_eigVect               # 低维特征空间的数据
    reconMat = (lowDDataMat*n_eigVect.T)+meanVal  # 重构数据
    print("lowDDataMat type :" + str(type(lowDDataMat)) + str(lowDDataMat.shape))
    return np.array(lowDDataMat)
