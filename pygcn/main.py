from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, output_view, plot_loss_with_acc_SF, plot_loss_with_acc_ER
from control import Cut_off_the_link, Isolation_link, Placing_correct_information_SF, Placing_correct_information_ER
from models import GCN

class gcnSetting():
    def __init__(self):
        # 训练设置
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='Disables CUDA training.')
        parser.add_argument('--fastmode', action='store_true', default=False,
                            help='Validate during training pass.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        # 完整训练次数
        parser.add_argument('--epochs', type=int, default=200,
                            help='Number of epochs to train.')
        # 学习率 学习率过小,收敛过慢，学习率过大,错过局部最优
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Initial learning rate.')
        # 权重衰减,正则化系数 weight_dacay，解决过拟合问题
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        self.args = parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # 载入数据
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data()

        # 定义模型与优化器
        self.model = GCN(nfeat=self.features[0][0].shape[1],
                    nhid=self.args.hidden,
                    nclass=int(self.labels[0][0].max().item() + 1),
                    dropout=self.args.dropout,
                    time_step=9)
        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=self.args.lr, weight_decay=self.args.weight_decay)

        # 是否使用GPU
        if self.args.cuda:
            self.model.cuda()
            print("using gpu")
            self.features = [f_time.cuda() for f_sample in self.features for f_time in f_sample]
            self.adj = self.adj.cuda()
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

        try:
            self.model.load_state_dict(torch.load("gcn_model.pth"))
        except FileNotFoundError:
            t_total = time.time()
            loss, val_acc = self.train() 
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            # 绘制SF数据集图像
            plot_loss_with_acc_SF(loss, val_acc)
            # 绘制ER数据集图像
            # plot_loss_with_acc_ER(loss, val_acc)

    def train(self):
        loss_history = []
        val_acc_history = []

        # Train model  逐个epoch进行train
        for sample in self.idx_train:
            for step in range(8):
                for epoch in range(self.args.epochs):
                    model_feature = self.features[sample*9+step]
                    model_label = self.labels[sample][step]

                    # 返回当前时间
                    t = time.time()
                    # 将模型转为训练模式，并将优化器梯度置零
                    self.model.train()
                    # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
                    # pytorch中每一轮batch需要设置optimizer.zero_grad
                    self.optimizer.zero_grad()
                    # 计算输出时，对所有的节点都进行计算
                    output = self.model(model_feature, self.adj)

                    # 损失函数，仅对训练集的节点进行计算，即：优化对训练数据集进行
                    loss_train = F.nll_loss(output, model_label.long())
                    # 计算准确率
                    acc_train = accuracy(output, model_label)
                    # debug
                    # output_view(sample, step, epoch, output, model_label)
                    # debug
                    # 反向求导  Back Propagation
                    loss_train.backward()
                    # 更新所有的参数
                    self.optimizer.step()
                    # 通过计算训练集损失和反向传播及优化，带标签的label信息就可以smooth到整个图上。

                    # 先是通过model.eval()转为测试模式，之后计算输出，并单独对测试集计算损失函数和准确率。
                    if not self.args.fastmode:
                        # 单独评估验证集性能，
                        # 在验证运行期间停用dropout。
                        # eval() 函数用来执行一个字符串表达式，并返回表达式的值
                        for val_sample in self.idx_val:
                            for val_step in range(8):
                                model_feature = self.features[val_sample * 9 + val_step]
                                model_label = self.labels[val_sample][val_step]
                                self.model.eval()
                                output = self.model(model_feature, self.adj)

                                # 验证集的损失函数
                                loss_val = F.nll_loss(output, model_label.long())
                                acc_val = accuracy(output, model_label)

                    # 记录训练过程中损失值和准确率的变化，用于画图
                    loss_history.append(loss_train.item())
                    val_acc_history.append(acc_val.item())
                    print('Sample:{:04d}'.format(sample),
                          'Step:{:02d}'.format(step),
                          'Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss_train.item()),
                          'acc_train: {:.4f}'.format(acc_train.item()),
                          'loss_val: {:.4f}'.format(loss_val.item()),
                          'acc_val: {:.4f}'.format(acc_val.item()),
                          'time: {:.4f}s'.format(time.time() - t))

        plot_loss_with_acc_ER(loss_history, val_acc_history)
        torch.save(self.model.state_dict(), "gcn_model.pth")
        return loss_history, val_acc_history

    def test(self, time_step):
        self.model.eval()
        output = self.model(self.features[time_step], self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[time_step][self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[time_step][self.idx_test])
        # 测试集的损失与精度
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        # 控制实验
        # control = output
        # 切断链路
        # Cut_off_the_link(control)
        # 隔离链路
        # Isolation_link(control)
        # 对SF数据集投放正确信息
        # Placing_correct_information_SF(control)
        # 对ER数据集投放正确信息
        # Placing_correct_information_ER(control)

    def use(self, feature, adj):
        self.model.eval()
        output = self.model(feature, adj)
        return output

if __name__ == '__main__':
    gcn = gcnSetting()