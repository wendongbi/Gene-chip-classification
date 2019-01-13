import matplotlib.pyplot as plt

def drawAccFig(filename, ti, le, picname):

    '''
    learning_rate = []
    plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow', 'pink', 'magenta', 'black', 'cyan'])
    for (key, value) in result.items():
        if best_reg in key:
            lr = key[0]
            learning_rate.append(lr)
            acc_his = value[0]
            plt.plot(acc_his)
    plt.plot()
    plt.legend(['lr = %.3f' % learning_rate[0], 'lr = %.3f' % learning_rate[1], 'lr = %.3f' % learning_rate[2],
                'lr = %.3f' % learning_rate[3], 'lr = %.3f' % learning_rate[4], 'lr = %.3f' % learning_rate[5],
                'lr = %.3f' % learning_rate[6], 'lr = %.3f' % learning_rate[7]], loc='lower right')
    '''
    aclist = []
    f = open(filename,'r')
    for i in range(100):
        aclist.append(float(f.readline()[:-1]))
    ep = [i+1 for i in range(100)]
    f.close()
    plt.plot(ep, aclist, 'r.-')
    plt.xlabel('epoch num')
    plt.ylabel('Training Accuracy')
    plt.xlim((0,100))
    plt.ylim((0,1))
    plt.title(ti)
    plt.legend(le)
    # plt.show()
    plt.savefig(picname)
    plt.show()

def drawAccFig2(fi1, fi2, ti, le, picname):
    aclist1 = []
    aclist2 = []
    f1 = open(fi1,'r')
    f2 = open(fi2, 'r')
    for i in range(100):
        aclist1.append(float(f1.readline()[:-1]))
        aclist2.append(float(f2.readline()[:-1]))

    ep = [i+1 for i in range(100)]
    f1.close()
    f2.close()
    plt.plot(ep, aclist1, 'r.-')
    plt.plot(ep, aclist2, 'g.-')
    plt.xlabel('epoch num')
    plt.ylabel('Training Accuracy')
    plt.xlim((0,100))
    plt.ylim((0,1))
    plt.title(ti)
    plt.legend(le)
    # plt.show()
    plt.savefig(picname)
    plt.show()


def drawAccFig3(fi1, fi2,fi3, ti, le, picname):
    aclist1 = []
    aclist2 = []
    aclist3 = []
    f1 = open(fi1, 'r')
    f2 = open(fi2, 'r')
    f3 = open(fi3, 'r')
    for i in range(100):
        aclist1.append(float(f1.readline()[:-1]))
        aclist2.append(float(f2.readline()[:-1]))
        aclist3.append(float(f3.readline()[:-1]))

    ep = [i + 1 for i in range(100)]
    f1.close()
    f2.close()
    f3.close()
    plt.plot(ep, aclist1, 'r.-')
    plt.plot(ep, aclist2, 'g.-')
    plt.plot(ep, aclist3, 'b.-')
    plt.xlabel('epoch num')
    plt.ylabel('Training Accuracy')
    plt.xlim((0, 100))
    plt.ylim((0, 1))
    plt.title(ti)
    plt.legend(le)
    # plt.show()
    plt.savefig(picname)
    plt.show()

if __name__ == '__main__':
    f1 = "LR/output.txt"
    f2 = "SVM/output.txt"
    f3 = "DNN/output.txt"
    drawAccFig(f1,"LR Binary Classifier", ["LR Binary"], "LR_Binary.png")
    drawAccFig(f2,"SVM Binary Classifier", ["SVM Binary"], "SVM_Binary.png")
    drawAccFig(f3,"DNN Binary Classifier", ["DNN Binary"], "DNN_Binary.png")
    f4 = "LR/output1.txt"
    f5 = "SVM/output1.txt"
    f6 = "DNN/output1.txt"
    drawAccFig(f4, "LR Multi Classifier", ["LR Multi"], "LR_Multi.png")
    drawAccFig(f5, "SVM Multi Classifier", ["SVM Multi"], "SVM_Multi.png")
    drawAccFig(f6, "DNN Multi Classifier", ["DNN Multi"], "DNN_Multi.png")

    drawAccFig2(f1,f4,"LR Classifier", ["LR Binary","LR Multi"],"LR.png")
    drawAccFig2(f2,f5, "SVM Classifier", ["SVM Binary", "SVM Multi"], "SVM.png")
    drawAccFig2(f3, f6, "DNN Classifier", ["DNN Binary", "DNN Multi"], "DNN.png")

    drawAccFig3(f1,f2,f3, "Three Binary Classifiers", ["LR Binary", "SVM Binary", "DNN Binary"], "Binary.png")
    drawAccFig3(f4, f5, f6, "Three Multi Classifiers", ["LR Multi", "SVM Multi", "DNN Multi"], "Multi.png")
