import numpy as np
import math

trX = []
trY = []

row = 22283
col = 5896
num = 0
maxN = 0
percent = 0

fx = open('../trans.txt','r')
fy = open('../regular.txt','w')
fx.readline()

for i in range(0,5896):
    line_x = fx.readline().split('\t')[:-1]
    #if float(line_x[1])<10:
     #       continue
    num+=1
    line_x = list(map(float,line_x))
    for j in line_x:
        if(abs(j) > abs(maxN)):
            maxN = j
    trX.append(line_x)
    if(i % 58 == 0):
        percent = percent + 1
        print("Input %:",percent)

print(num)
print(maxN)
print (len(trX))
for i in range(0,len(trX)):
    for j in range(0,len(trX[i])):
        fy.write(str(trX[i][j]/maxN)+'\t')
    fy.write('\n')
    if(i % 58 == 0):
        percent = percent + 1
        print("Output %:",percent)
    
