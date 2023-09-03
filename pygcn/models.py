"""

定义GCN模型，即用预先定义的图卷积层来组建GCN模型。

此部分与pytorch中构建经典NN模型的方法一致。

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import layers


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, time_step):
        """
           参数1 ：nfeat   输入层数量，即特征feature的维度
           参数2： nhid    输出特征数量，隐藏层的隐藏单元
           参数3： nclass  输出层单元数即分类的类别数
           参数4： dropout dropout概率
        """
        super(GCN, self).__init__()
        # 第一层GCN
        self.gc1 = layers.GraphConvolution(nfeat, nhid)
        # 第二层GCN
        self.gc2 = layers.GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.time_step = time_step

    # 定义前向计算，即各个图卷积层之间的计算逻辑
    def forward(self, x, adj):
        # 第一层的输出
        first = F.relu(self.gc1(x, adj))
        curr_step = F.dropout(first, self.dropout, training=self.training)
        # 第二层的输出
        second = self.gc2(curr_step, adj)
        res = F.log_softmax(second, dim=1)
        return res
