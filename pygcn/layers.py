"""
定义图卷积层，图卷积层包括两个操作：邻居聚合与特征变换。
-邻居聚合：用于聚合邻居结点的特征。
-特征变换：传统NN的操作，即特征乘参数矩阵。

基于pytorch实现，需要导入torch中的parameter和module模块。
"""

import math

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# 定义图卷积层
class GraphConvolution(Module):
    # 图卷积层的作用是接收旧特征并产生新特征
    # 因此初始化的时候需要确定两个参数：输入特征的维度与输出特征的维度
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 由于weight（权重）是可以训练的，因此使用parameter定义
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 由于bias（偏移向量）是可以训练的，因此使用parameter定义
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 参数初始化
    def reset_parameters(self):
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform() 方法将随机生成下一个实数，它在 [x, y] 范围内
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 输入是旧特征+邻接矩阵,输出是新特征
    def forward(self, input, adj):
        """
        参数input：表示输入的各个节点的特征矩阵
        参数adj ：表示邻接矩阵
        """
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        # torch.spmm(a,b)是稀疏矩阵相乘
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
