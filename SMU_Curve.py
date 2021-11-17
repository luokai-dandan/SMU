# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from SMU import *

def Pretreatment(x, alpha:float, mu:list, reluFlag=True) -> list:
    smu0 = SMU(alpha=alpha, mu=mu[0])
    y0 = smu0(x)

    smu1 = SMU(alpha=alpha, mu=mu[1])
    y1 = smu1(x)

    smu2 = SMU(alpha=alpha, mu=mu[2])
    y2 = smu2(x)

    if reluFlag:
        leakyRelu = torch.nn.ReLU()
        y3 = leakyRelu(x)
    else:
        leakyRelu = torch.nn.LeakyReLU(alpha)
        y3 = leakyRelu(x)

    # 类型转换(带梯度的Tensor->Numpy)
    result = []
    result.append(x.detach().numpy())
    result.append(y0.detach().numpy())
    result.append(y1.detach().numpy())
    result.append(y2.detach().numpy())
    result.append(y3.detach().numpy())

    # [x, y0, y1, y2, y3]
    return result


def Curve01(x, alpha:float, mu:list, reluFlag=True):
    # alpha = 0.0
    # mu = [1.0, 100, 0.5]
    [x, y0, y1, y2, y3] = Pretreatment(x, alpha=alpha, mu=mu, reluFlag=reluFlag)

    plt.title('SMU α=0')
    plt.plot(x, y0, color='skyblue', label='SMU,μ=1.0')
    plt.plot(x, y1, color='red', label='SMU,μ=100.0')
    plt.plot(x, y2, color='green', label='SMU,μ=0.5')
    plt.plot(x, y3, '--', color='blue', label='ReLU')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def Curve02(x, alpha:float, mu:list, reluFlag=True):
    # alpha = 0.25
    # mu = [1.0, 100, 0.5]
    [x, y0, y1, y2, y3] = Pretreatment(x, alpha=alpha, mu=mu, reluFlag=reluFlag)

    # 绘图
    plt.title('SMU α=0.25')
    plt.plot(x, y0, color='skyblue', label='SMU,μ=1.0')
    plt.plot(x, y1, color='red', label='SMU,μ=100.0')
    plt.plot(x, y2, color='green', label='SMU,μ=0.5')
    plt.plot(x, y3, '--', color='blue', label='Leaky ReLU')
    plt.legend()
    # 加网格
    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    x = torch.linspace(-5, 5, 1000)
    mu = [1.0, 100, 0.5]
    Curve01(x, alpha=0.0, mu=mu, reluFlag=True)
    Curve02(x, alpha=0.25, mu=mu, reluFlag=False)
