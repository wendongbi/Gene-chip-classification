#coding=utf-8

# Copyright: Copyright (c) 2018

# All rights reserved

# Created on 2018-12-18  

# Author:Wendong Bi

# Filename: comp2float.py

# discription:
# 经过PCA降维后的数据是以复数的形式输出为txt文件的，即：(float+0j)的形式，而我们要的形式是float因此
# 需要去掉每个float两边的( +0j)



import os
import io



def alter(file,old_str,new_str):

    """

    替换文件中的字符串

    :param file:文件名

    :param old_str:就字符串

    :param new_str:新字符串

    :return:

    """

    file_data = ""

    with io.open(file, "r", encoding="utf-8") as f:

        for line in f:

            if old_str in line:

                line = line.replace(old_str,new_str)

            file_data += line

    with io.open(file,"w",encoding="utf-8") as f:

        f.write(file_data)

file = 'data_dimRed_0.9.txt'
old_str1 = '('
old_str2 = '+0j)'

alter(file, old_str1, '')
alter(file, old_str2, '')

# #获取目录下的文件

# def file_name(file_dir):   

# 	for root, dirs, files in os.walk(file_dir): 

# 		return (files)

 

# #获取后缀名

# def file_extension(file): 

#   return os.path.splitext(file)[1]

 

 

# file_dir='./'

# file_list = file_name(file_dir)

 

 

# for i in file_list:

# 	if file_extension(i)==".ctl":

# 		alter(i,'ZHS16GBK','AL32UTF8')

