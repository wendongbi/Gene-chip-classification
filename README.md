# AI-Project-2018-
-----------------------------------------
AI Project(2018) About Genechip information classification
lib：Tensorflow, numpy, keras...
language:python.

##下面解释每个文件的作用：
comp2float.py：因为用numpy实现PCA之后生成的数据全是复数的形式，然后通过这个文件把它们变为float。
 
data_dimRed_0.9.txt ， 
data_dimRed_0.99.txt ，
data_dimRed_0.95.txt 分别是用PCA降维成不同维度的数据集。

model文件夹内为所有实现的模型代码：包括LR,DNN,SVM,KNN等。可视化有python中matplotlib画的图的代码和tensorboard（以写在代码当中）
labelData文件夹内为处理的数据集和标注的label等。
