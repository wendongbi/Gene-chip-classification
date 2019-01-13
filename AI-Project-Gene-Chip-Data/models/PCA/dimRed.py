#!/usr/env/bin
#-*-coding:utf-8 -*-

# Copyright: Copyright (c) 2018

# All rights reserved

# Created on 2018-12-18  

# Author:Wendong Bi

# Filename: svm_Muti_diseasestate.py

# batch_pca_process, process the raw data to the low dimentions with file pca.py

import numpy as np
import string
from pca import *
import time
if __name__ == '__main__':
    start_time = time.time()
    sourcefile = 'F:/ai_data/Gene_Chip_Data/microarray.original.txt'
    f = open(sourcefile,'r')
    f.readline()

    print ("Now reading data...")

    ######test######

    # num = 0
    # for i in range(0,22283):
    #     line = f.readline()
    #     line = line[:-1].split('\t')
    #     # line = map(string.atof,line[1:])
    #     if string.atof(line[1])>=5:
    #         num+=1
    ######test######
    PCA_percentage = 0.95
    k_dimentions = 453
    num = 0
    data = []
    for i in range(0, 22283):
        line = f.readline()
        line = line[:-1].split('\t')
        if i % 200 == 0:
            print(i/200)
        if float(line[1]) < 10:  # 第一个数字<10的直接省略这一行特征，手动降维，如果电脑内存足够去掉此行对全部维度进行pca降维
            continue
        num += 1  # 最后选出来的列数代表维度（根据原始数据p*n
        line = line[1:]    # 去掉第一个字符串
        line = list(map(float, line))
        data.append(line)

    f.close()
    print("Data has been read successfully.")
    print("The dimension is "+str(num))
    print(data[0][0])

    data = np.array(data).T     # 原始数据一列代表一个example，一行代表一个维度，现在要求转置，变成可以求PCA的形式，也就是一行代表一个example，一列代表一个维度。
    print("Now reducing dimension...")
    lowDData = pca(dataMat=data, percentage=0, k=k_dimentions)
    print("Finished, the new dimension is :"+str(len(lowDData[0])))

    print("Start writing new data...")
    destfile = '../../data_dimRed_' + str(PCA_percentage) + '.txt'
    print(len(lowDData))
    f = open(destfile,'w')
    for i in range(0,len(lowDData)):
        for j in range(0,len(lowDData[i])):
            f.write(str(lowDData[i][j])+'\t')
        f.write('\n')
    end_time = time.time()
    duration = end_time - start_time
    print('Time cost: %fs.\n' % float(duration))
    print ("Finished the whole work.")

    # testArray = np.array([[4,3,2],[3,2,1],[2,0,0]])
    # lowd,res = pca(testArray)
    # print res
