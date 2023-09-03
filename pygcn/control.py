import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as scio


def Cut_off_the_link(output):
    """切断链路"""
    # 将预测的传播源概率最大的一百个节点及其二阶相邻节点的传播路径切断，阻止这些节点的传染。
    # mum不控制感染节点数，num该控制方法感染节点数，aum控制前节点数

    path = r'../data/cora/SF.mat'
    data = scio.loadmat(path)
    res = data['res900']
    adj = data['aw']
    features = data['tz']
    num = 0
    mum = 0
    aum = 0
    max10 = []

    for j in range(1000):
        if res[j, 0] != 0:
            mum = mum + 1

    x = output[:, 1]
    x = x.sort()[1]
    for i in range(100):
        max10.append(x[999 - i].item())

    for m in range(100):
        mark = max10[m]
        ori = []
        ori2 = []
        ori3 = []
        for i in range(1000):
            if adj[mark, i] == 1:
                if features[i, 0] == 0:
                    res[j, 0] = 0
                ori.append(i)
        for i in range(len(ori)):
            for j in range(1000):
                if adj[ori[i], j] == 1:
                    ori2.append(j)
                    if features[j, 0] == 0:
                        res[j, 0] = 0
        for i in range(len(ori2)):
            for j in range(1000):
                if adj[ori2[i], j] == 1:
                    ori3.append(j)
                    if features[j, 0] == 0:
                        res[j, 0] = 0
        for i in range(len(ori3)):
            for j in range(1000):
                if adj[ori3[i], j] == 1:
                    if features[j, 0] == 0:
                        res[j, 0] = 0

    for j in range(1000):
        if res[j, 0] != 0:
            num = num + 1

    for j in range(1000):
        if features[j, 0] != 0:
            aum = aum + 1

    print("不控制方法感染节点总数为：", mum)
    print("切断链路感染节点总数为：", num)
    print("原感染节点总数为：", aum)
    print("切断链路感染程度为：", num / 1000)
    print("切断链路新增感染节点数为：", num - aum)
    print("切断链路控制效率为：", format((mum - aum) / (num - aum), '.3f'))
    # return res


def Isolation_link(output):
    """隔离链路"""
    # 将预测的传播源概率最大的一百个节点所在的区域进行隔离。
    # 仅允许区域内的节点传播，但阻止区域间与区域外的信息传播和节点跨区域的信息传播。
    # mum不控制感染节点数，num该控制方法感染节点数，aum控制前节点数

    path = r'../data/cora/SF.mat'
    data = scio.loadmat(path)
    res = data['res900']
    adj = data['aw']
    features = data['tz']
    area = features[:, 0]
    num = 0
    mum = 0
    aum = 0
    max10 = []

    for j in range(1000):
        if res[j, 0] != 0:
            mum = mum + 1

    for j in range(1000):
        if features[j, 0] != 0:
            aum = aum + 1

    x = output[:, 1]
    x = x.sort()[1]
    for i in range(100):
        max10.append(x[999 - i].item())

    for m in range(100):
        mark = max10[m]
        ori = []
        ori2 = []
        ori3 = []
        for i in range(1000):
            if adj[mark, i] == 1:
                if res[i, 0] > 0:
                    area[j] = 1
                ori.append(i)
        for i in range(len(ori)):
            for j in range(1000):
                if adj[ori[i], j] == 1:
                    ori2.append(j)
                    if res[j, 0] > 0:
                        area[j] = 1
        for i in range(len(ori2)):
            for j in range(1000):
                if adj[ori2[i], j] == 1:
                    ori3.append(j)
                    if res[j, 0] > 0:
                        area[j] = 1
        for i in range(len(ori3)):
            for j in range(1000):
                if adj[ori3[i], j] == 1:
                    if res[j, 0] > 0:
                        area[j] = 1

    for j in range(1000):
        if area[j] != 0:
            num = num + 1

    print("不控制方法感染节点总数为：", mum)
    print("隔离链路感染节点总数为：", num)
    print("原感染节点总数为：", aum)
    print("隔离链路感染程度为：", num / 1000)
    print("隔离链路新增感染节点数为：", num - aum)
    print("隔离链路控制效率为：", format((mum - aum) / (num - aum), '.3f'))
    # return res


def Placing_correct_information_SF(output):
    """对SF数据集投放正确信息"""
    # 对预测的传播源概率最大的一百个节点及其二阶相邻节点实行正确信息投放。
    # 根据网络数据集，直接修改节点的信息以使其不再呈现感染状态，并免疫之后的不良传染，但不阻止其余节点的信息传播。
    # mum不控制感染节点数，num该控制方法感染节点数，aum控制前节点数

    path = r'../data/cora/SF.mat'
    data = scio.loadmat(path)
    res = data['res900']
    adj = data['aw']
    features = data['tz']
    num = 0
    mum = 0
    aum = 0

    max10 = []

    for j in range(1000):
        if res[j, 0] != 0:
            mum = mum + 1

    x = output[:, 1]
    x = x.sort()[1]
    for i in range(100):
        max10.append(x[999 - i].item())

    for m in range(100):
        mark = max10[m]

        ori = []
        ori2 = []
        res[mark, 0] = 0
        for i in range(1000):
            if adj[mark, i] == 1:
                res[i, 0] = 0
                ori.append(i)
        for i in range(len(ori)):
            for j in range(1000):
                if adj[ori[i], j] == 1:
                    ori2.append(j)
                    if features[j, 0] == 0:
                        res[j, 0] = 0
        for i in range(len(ori2)):
            for j in range(1000):
                if adj[ori2[i], j] == 1:
                    if features[j, 0] == 0:
                        res[j, 0] = 0

    for j in range(1000):
        if res[j, 0] > 0:
            num = num + 1

    for j in range(1000):
        if features[j, 0] != 0:
            aum = aum + 1

    print("不控制方法感染节点总数为：", mum)
    print("投放正确信息感染节点总数为：", num)
    print("原感染节点总数为：", aum)
    print("投放正确信息感染程度为：", num / 1000)
    print("投放正确信息新增感染节点数为：", num - aum)
    print("投放正确信息控制效率为：", format((mum - aum) / (num - aum), '.3f'))
    # return res


def Placing_correct_information_ER(output):
    """对ER数据集投放正确信息"""
    # 对预测的传播源概率最大的一百个节点及其二阶相邻节点实行正确信息投放。
    # 根据网络数据集，直接修改节点的信息以使其不再呈现感染状态，并免疫之后的不良传染，但不阻止其余节点的信息传播。
    # mum不控制感染节点数，num该控制方法感染节点数，aum控制前节点数

    path = r'../data/cora/ER.mat'
    data = scio.loadmat(path)
    res = data['res900']
    adj = data['aw']
    features = data['tz']
    num = 0
    mum = 0
    aum = 0

    max10 = []

    for j in range(1000):
        if res[j, 0] != 0:
            mum = mum + 1

    x = output[:, 1]
    x = x.sort()[1]
    for i in range(10):
        max10.append(x[999 - i].item())

    for m in range(10):
        mark = max10[m]

        ori = []
        res[mark, 0] = 0
        for i in range(1000):
            if adj[mark, i] == 1:
                res[i, 0] = 0
                ori.append(i)
        for i in range(len(ori)):
            for j in range(1000):
                if adj[ori[i], j] == 1:
                    if features[j, 0] == 0:
                        res[j, 0] = 0

    for j in range(1000):
        if res[j, 0] > 0:
            num = num + 1

    for j in range(1000):
        if features[j, 0] != 0:
            aum = aum + 1

    print("不控制方法感染节点总数为：", mum)
    print("投放正确信息感染节点总数为：", num)
    print("原感染节点总数为：", aum)
    print("投放正确信息感染程度为：", num / 1000)
    print("投放正确信息新增感染节点数为：", num - aum)
    print("投放正确信息控制效率为：", format((mum - aum) / (num - aum), '.3f'))
    # return res
