from sklearn import decomposition
import sys
import csv

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def normalize(x):
    mean = x.mean()
    std = x.std()
    for xx in x:
        xx = (xx - mean) / std

    return x

def roundRobin(l, turn):
    for i in range(int(len(l)/turn)):
        l.append(l[i])
        del l[i]

def setList(index, labelNum):
    l = []
    for i in range (labelNum):
        if i == index:
            l.append(1)
        else:
            l.append(0)
    return l

def findIndex(list, element):
    for i in list:
        if element == i:
            return list.index(i)
    return -1

# pca feature dimension
if len(sys.argv) != 1:
    ndim = int(sys.argv[1])
# label dimension
ldim = 1
batch_size = 100
nb_epoch = 100
dropout_rate = 0.5
hidden_width = 512

numTrain = 5896
numTest = 5896
iteration = 100
robin = 10
batch = 100
rate = 0.05
errorTrain = 0
errorTest = 0

trX = []
trY = []
# train
fy = open('../../E-TABM-185.sdrf.txt', 'r')
fx = open('../../data_dimRed_0.9.txt', 'r')
fo = open('output.txt', 'w')
fy.readline()
errorTrain = 0
suit_label = []
useful_label = []
finallist = []
with open('../../labelDetail/2_Material Type.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if int(row[1]) < 10:
            suit_label.append(row[0])

for i in suit_label:
    print(i)

for i in range(0, 5896):
    line_x = fx.readline().split('\t')[:-1]
    line_x = list(map(float, line_x))
    # if(i%300 == 0):
    #   print(len(line_x))
    line_y = fy.readline().split('\t')[1:2][0]
    if line_y in suit_label:
        continue
    else:
        if line_y[0:2] == "  ":
            #invalid_cnt = invalid_cnt + 1
            continue
        trX.append(line_x)
        ret = findIndex(useful_label, line_y)
        if line_y in suit_label:
            continue
        else:
            if line_y == 'organism_part':
                line_y = 1
            else:
                line_y = 0
            trY.append(line_y)

ndim = len(trX[0])
order = np.random.permutation(len(trX))
order = order.tolist()
tpx = [trX[i] for i in order]
tpy = [trY[i] for i in order]
trX = tpx
trY = tpy

dtlist = []
for i in range(iteration):
    dtlist.append(0)
for turn in range(10):
    roundRobin(trX, robin)
    roundRobin(trY, robin)

    teX = np.array(trX[0:round(len(trX) / robin)]).astype('float32')
    teY = np.array(trY[0:round(len(trY) / robin)]).astype('float32')

    tpx = np.array(trX[int(len(trX) / robin):]).astype('float32')
    tpy = np.array(trY[int(len(trX) / robin):]).astype('float32')

    #teX = np_utils.to_categorical(teX, ldim)
    #teY = np_utils.to_categorical(teY, 1)
    #tpx = np_utils.to_categorical(tpx, ldim)
    #tpy = np_utils.to_categorical(tpy, 1)



    # normalization to mean=0, var=1
    # X_train = normalize(X_train)
    # X_vali = normalize(X_vali)
    # X_test = normalize(X_test)

    # writ("train.txt", X_train)
    # writ("vali.txt", X_vali)
    # writ("test.txt", X_test)
    #
    # X_train = load("./train.txt").astype('float32')
    # X_vali = load("./vali.txt").astype('float32')
    # X_test = load("./test.txt").astype('float32')

    print("Network building begin")
    model = Sequential()
    model.add(Dense(hidden_width, input_shape=(ndim,), activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_width, activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_width, activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    # print model structure
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # estimator = KerasClassifier(build_fn=model, nb_epoch=30, batch_size=64)

    # seed = 100
    # np.random.seed(seed)
    # kfold = KFold(n_splits=10, shuffle=True)
    # results = cross_val_score(estimator, trX, trY, cv=kfold)

    print
    "Network training begin"
    History = model.fit(tpx, tpy, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                        validation_data=(teX, teY))
    acc = History.history['acc']
    for item in range(100):
        dtlist[item]+=acc[i]
dtlist = [item / 10 for item in dtlist]
for item in dtlist:
    fo.write(str(item) + '\n')
print (sum(dtlist)/len(dtlist))