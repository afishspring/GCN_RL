import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

import numpy as np
import torch
import sys
import random
sys.path.append("../pygcn")

from pygcn import gcnSetting, load_data, tensor_adj

class rumor(gym.Env):
    def __init__(self):
        self.model = gcnSetting()

        adj = self.model.adj
        features = self.model.features

        self.adj_raw = adj.to_dense().cpu().numpy()
        self.adj = adj.to_dense().cpu().numpy()

        self.time_step = 9
        self.seed = random.randint(0, int(len(features) / self.time_step))
        self.feature_raw = features[self.seed * self.time_step]
        self.feature = features[self.seed * self.time_step]

        # 初始化环境参数
        self.num_nodes = self.feature[:,0].shape[0]
        self.num_edges = self.num_nodes * self.num_nodes
        self.choose_space = 15
        self.num_actions = 3 * self.choose_space

        # 定义动作空间
        self.action_space = spaces.Discrete(self.num_actions)

        # 定义状态空间（简化为每个节点的状态）
        self.observation_space = spaces.MultiBinary(self.num_nodes)

        # 初始化环境
        self.reset()

    def reset(self):
        # 随机初始化节点状态
        self.node_states = self.feature_raw[:, 0].cpu().numpy()

        # 随机初始化图的连接情况
        self.graph = self.adj_raw

        # 将图设为对称（无向图）
        self.graph = np.logical_or(self.graph, self.graph.T).astype(int)
        np.fill_diagonal(self.graph, 1)

        # 记录已执行的步数
        self.steps = 0

        return self.node_states

    def step(self, action):
        output = self.model.use(self.feature, tensor_adj(self.adj).cuda())
        node_list = np.argsort(output[:, self.steps+1].cpu().detach().numpy())[:self.choose_space]


        if action < self.choose_space:
            # 切断链路
            reward = -1
        elif action < 2 * self.choose_space:
            # 隔离节点
            reward = 0
        elif action < 3 * self.choose_space:
            # 投放正确信息
            for node in node_list:
                self.node_states[node] = 0
            reward = 0
        else:
            reward = 0

        # if action < self.num_nodes:
        #     self.node_states[action] = 0
        # else:
        #     # 边操作
        #     edge_idx = action - self.num_nodes
        #     row = edge_idx // self.num_nodes
        #     col = edge_idx % self.num_nodes
        #     if row!=col:
        #         self.graph[row, col] = False
        #         self.graph[col, row] = False
        #
        # for i in range(self.num_nodes):
        #     self.feature[i, 0] = self.node_states[i].item()
        # feature = self.feature
        # adj = tensor_adj(self.graph)
        # output = self.model.use(feature, adj)
        # # 计算奖励
        # reward = -np.sum(self.node_states)  # 节点值为0的越多，奖励越高

        # 更新步数
        self.steps += 1

        # 判断是否结束
        done = self.steps >= 9

        # 返回状态、奖励、是否结束的信息
        return self.node_states, reward, done, {}

    def updateGraph(self):
        for i in range(self.num_nodes):
            self.feature[i, 0] = self.node_states[i].item()
        feature = self.feature
        adj = tensor_adj(self.graph).cuda()

    def render(self):
        # 这里可以定义可视化的内容（可选）
        pass