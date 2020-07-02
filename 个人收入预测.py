import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1 import host_subplot

def train(x_train, y_train, state ,losses ,accs):
    num = x_train.shape[0]
    dim = x_train.shape[1]
    bias = 0  # 偏置值初始化
    weights = np.ones(dim)  # 权重初始化
    learning_rate = 0.1  # 学习率
    reg_rate = 0.1  # 正则项系数
    bg2_sum = 0  # 偏置值
    wg2_sum = np.zeros(dim)  # 权重


    for i in range(state):
        b_g = 0
        w_g = np.zeros(dim)

        for j in range(num):
            y_pre = weights.dot(x_train[j, :]) + bias
            sig = 1 / (1 + np.exp(-y_pre))
            b_g += (-1) * (y_train[j] - sig)
            for k in range(dim):
                w_g[k] += (-1) * (y_train[j] - sig) * x_train[j, k] + 2 * reg_rate * weights[k]

        b_g /= num
        w_g /= num

        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2

        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        if i!=0:
            loss = 0
            acc = 0
            result = np.zeros(num)
            for j in range(num):
                y_pre = weights.dot(x_train[j, :]) + bias
                sig = 1 / (1 + np.exp(-y_pre))
                if sig >= 0.5:
                    result[j] = 1
                else:
                    result[j] = 0

                if result[j] == y_train[j]:
                    acc += 1.0
                #loss += math.fabs( y_train[j]-sig )
                loss += (-1) * (y_train[j] * np.log(sig+0.000000001) + (1 - y_train[j]) * np.log(1 - sig+0.000000001))
            if i>=1:
                losses[i-1] = loss / num
                accs[i-1] = acc / num
            print('after {} epochs, the loss on train data is:'.format(i), loss / num)
            print('after {} epochs, the acc on train data is:'.format(i), acc / num)


    return weights, bias


# 验证模型效果
def validate(x_val, y_val, weights, bias):
    num = 1000
    loss = 0
    acc = 0
    result = np.zeros(num)
    for j in range(num):
        y_pre = weights.dot(x_val[j, :]) + bias
        sig = 1 / (1 + np.exp(-y_pre))
        if sig >= 0.5:
            result[j] = 1
        else:
            result[j] = 0

        if result[j] == y_val[j]:
            acc += 1.0
        #loss += math.fabs(y_val[j] - sig)
        loss += (-1) * (y_val[j] * np.log(sig+0.000000001) + (1 - y_val[j]) * np.log(1 - sig+0.000000001))
    return acc / num


def main():
    df = pd.read_csv('2333.csv') #读取数据
    array = np.array(df)
    x = array[:, 1:-1]

    x[:, -1] /= np.mean(x[:, -1])#取平均值，使后三列数据也分布在1附近
    x[:, -2] /= np.mean(x[:, -2])
    x[:, -3] /= np.mean(x[:, -3])
    y = array[:, -1]

    x_train, x_val = x[0:3000, :], x[3000:4000, :]#各项熟悉
    y_train, y_val = y[0:3000], y[3000:4000]#是否大于50k



    state = 10 # 迭代次数

    losses = np.zeros(state-1)
    accs = np.zeros(state-1)

    w, b = train(x_train, y_train, state ,losses ,accs)# 用前4000行做训练

    acc = validate(x_val, y_val, w, b)  #在后1000行验证
    print('The acc on val data is:', acc)

    x1=range(1,state)
    y1=accs
    y2=losses

    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    par1 = host.twinx()

    host.set_xlabel("steps")
    host.set_ylabel("test-loss")
    par1.set_ylabel("test-accuracy")
    p1, = host.plot(x1 ,y2, label="loss")
    p2, = par1.plot(x1, y1, label="accuracy")

    host.legend(loc=5)

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.show()




if __name__ == '__main__':
    main()