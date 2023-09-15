import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import os
import networkx as nx
def raw_data(base_path):
    num_samples = 1000
    num_times = range(100, 1000, 100)
    num_cols = 1000

    filename = f'AdjMw.mat'
    filepath = os.path.join(base_path, filename)
    mat_data = scio.loadmat(filepath)
    aw = mat_data['Aw']

    result_array = np.zeros((num_samples, num_cols, len(num_times)))
    for i in range(1, num_samples + 1):
        for j_idx, j in enumerate(num_times):
            filename = f'Sstate_{i}_{j}.mat'
            filepath = os.path.join(base_path, filename)
            mat_data = scio.loadmat(filepath)
            data = mat_data['StateS']
            result_array[i - 1, :, j_idx] = data

    features_np = []
    for index, result in enumerate(result_array):
        if index > 9:
            break
        print("sample" + str(index))
        f_sample = []
        for time in range(9):
            f = eigenmatrixing(result[:, time], aw)
            f_sample.append(f)
        features_np.append(f_sample)

    labels = []
    for sample_idx in range(num_samples):
        sample_label = []
        for node_idx in range(num_cols):
            found_positive = False
            for time_idx in range(len(num_times)):
                value = result_array[sample_idx, node_idx, time_idx]
                if value > 0:
                    sample_label.append(time_idx)
                    found_positive = True
                    break
            if not found_positive:
                sample_label.append(10)  # 节点在所有时间步都保持为零
        labels.append(sample_label)

    labels_array = np.array(labels)
    return features_np, labels_array, aw
def load_data():
    """Load citation network dataset (cora only for now)"""
    base_path = r"../data/ER_75_sig_gen_1_ini"

    saved_mat = r"../data/cora/ER_gcn.mat"
    if os.path.exists(saved_mat):
        print("load exist data")
        data = scio.loadmat(saved_mat)
        Aw = data['Aw']
        labels_array = data['labels']
        features_np = data['features']
    else:
        print("load new data")
        features_np, labels_array, Aw = raw_data(base_path)
        data = {
            'Aw': Aw,
            'labels': labels_array,
            'features': features_np
        }
        scio.savemat(saved_mat, data)

    labels = []
    for label in labels_array:
        label = torch.from_numpy(label)
        label = label.to(torch.int64).squeeze()
        labels.append(label)
    features = []
    for f_s in features_np:
        f_sample = []
        for f in f_s:
            # 构造稀疏矩阵
            f_mat = sp.csr_matrix(f, dtype=np.float32)
            # 将features转化成torch稠密矩阵
            f_mat = torch.FloatTensor(np.array(f_mat.todense()))
            f_sample.append(f_mat)
        features.append(f_sample)

    adj = tensor_adj(Aw)

    n_sample = len(features)
    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的tensor，用来做模型的输入。
    idx_train = range(int(n_sample*0.6))
    idx_val = range(int(n_sample*0.6), int(n_sample*0.8))
    idx_test = range(int(n_sample*0.8), n_sample)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def tensor_adj(_adj):
    adj = sp.coo_matrix(_adj, dtype=np.float32)
    # 建立对称邻接矩阵。上一步得到的adj是按有向图构建的，将它转换成无向图的邻接矩阵需要扩充成对称矩阵。
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # features = normalize(features)
    # 对邻接矩阵adj做标准化，用到了normalize()方法
    # sp.eye 生成单位矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # 将scipy稀疏矩阵转换为torch稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize(mx):
    """行归一化稀疏矩阵"""
    # 首先对每一行求和得到rowsum；求倒数得到r_inv；如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0；
    # 构建对角元素为r_inv的对角矩阵；用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘。
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    # numpy中的ndarray转化成pytorch中的tensor: torch.from_numpy()
    # pytorch中的tensor转化成numpy中的ndarray: numpy()
    # .tocoo()将稠密矩阵转为稀疏矩阵
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    """精度计算函数"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def plot_loss_with_acc_SF(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = ax1.twinx()
    # ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('SF Training Loss & Validation Accuracy')
    plt.show()


def plot_loss_with_acc_ER(loss_history, val_acc_history):
    fig = plt.figure()
    # 坐标系ax1画曲线1
    ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c为颜色
    plt.ylabel('Loss')

    # 坐标系ax2画曲线2
    ax2 = ax1.twinx()
    # ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # 开启右边的y坐标

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('ER Training Loss & Validation Accuracy')
    plt.show()


def eigenmatrixing(res, adj):
    G = nx.Graph()
    # 添加节点
    for i in range(len(adj)):
        G.add_node(i)
    # 添加边
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            if adj[i][j] > 0:
                G.add_edge(i, j)

    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    """提取网络拓扑的特征"""
    features = np.zeros((1000, 5))
    # 节点感染状态
    for i in range(1000):
        features[i, 0] = 1 if res[i] > 0 else 0

    # 已感染邻居数量与未感染邻居数量
    for i in range(1000):
        res1 = 0
        res2 = 0
        for j in range(1000):
            if adj[i, j] != 0:
                if res[j] == 0:
                    res2 = res2 + 1
                else:
                    res1 = res1 + 1
        features[i, 1] = res1
        features[i, 2] = res2

    # 节点中心程度
    for i in range(1000):
        features[i, 3] = betweenness_centrality[i]
        features[i, 4] = closeness_centrality[i]
    return features
