# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-18
# Description:
# -----------------------------------------------------------------------#
import json

from knlp.common.constant import KNLP_PATH


class Inference:
    """
    hmm 的本质便是利用之前基于统计数据计算出来的几个概率，针对输入的sequence进行正向计算，以得到想要的结果
    应该具备的功能点
    1 load 模型和


    首先需要梳理，拿到一个sentence之后，要怎么计算获取到最终的分词结果。


    然后再考虑使用viterbi进行


    最后再想想怎么进行新词发现

    最后整体上过一遍工程，把相关的内容整理成文案，发到知乎，并发布在各个群里面，还有公众号里面。
    晚上要去公司，一方面是走走，另一方面是把之前的工作整理一下，把上周遗留下来的工作完成

    2 解码
    3 分词 输出结果

    """

    def __init__(self):
        self._state_set = {}
        self._transition_pro = {}
        self._emission_pro = {}
        self._init_state_set = {}
        self.load_mode()

    def load_mode(self, state_set=None, transition_pro=None, emission_pro=None, init_state_set=None):
        def helper(file_path, save_format="json"):
            if save_format == "json":
                with open(file_path, "w") as f:
                    return json.load(f)

        state_set = KNLP_PATH + "/knlp/model/state_set.json" if not state_set else state_set
        transition_pro = KNLP_PATH + "/knlp/model/transition_pro.json" if not transition_pro else transition_pro
        emission_pro = KNLP_PATH + "/knlp/model/emission_pro.json" if not emission_pro else emission_pro
        init_state_set = KNLP_PATH + "/knlp/model/init_state_set.json" if not init_state_set else init_state_set
        self._state_set = helper(file_path=state_set)
        self._transition_pro = helper(file_path=transition_pro)
        self._emission_pro = helper(file_path=emission_pro)
        self._init_state_set = helper(file_path=init_state_set)

    def viterbi(self, observe_seq, state_set=None, init_state_set=None, transition_pro=None, emission_pro=None):
        if not state_set:
            state_set = self._state_set
        if not init_state_set:
            init_state_set = self._init_state_set
        if not transition_pro:
            transition_pro = self._transition_pro
        if not emission_pro:
            emission_pro = self._emission_pro
        V = [{}]  # 每个timestep的几个概率大小，数组的index为timestep，里面的字典为概率值。可以想象为一个矩阵，横轴为timestep，纵轴为不同的概率值
        path = {}  # key 是当前的使整体概率最大的hidden state，value是一个数组，保存了路由到当前这个hidden state的，之前的所有的hidden state
        hidden_state_set = state_set['hidden_state']
        # 计算初始状态的概率分布，以及对应的路径， timestep = 1
        for hidden_state in hidden_state_set:
            V[0][hidden_state] = init_state_set[hidden_state] * emission_pro[hidden_state].get(observe_seq[0], 0)
            path[hidden_state] = hidden_state
        # 计算后续几个状态的分布以及对应的路径，timestep > 1
        for timestep in range(1, len(observe_seq)):
            V.append({})
            new_path = {}
            for hidden_state in hidden_state_set:  # 循环遍历这个timestep可以采用的每一个hidden state
                # 这里这个就是我们求解viterbi算法需要使用的公式
                # 这里做的就是将上一个timestep的每个状态到达当前的状态的概率大小计算了一下，得到最大的那个状态以及对应的概率值
                # 注意max在对数组处理的时候会用数组的第一个元素进行处理，如果第一个值一样，会用第二个进行
                max_prob, arg_max_prob_hidden_state = max([(
                    V[timestep - 1][hidden_state0] * transition_pro[hidden_state0].get(hidden_state, 0) * emission_pro[
                        hidden_state].get(observe_seq[timestep], 0), hidden_state0)] for
                                                          # 这个for循环 循环的是前一个timestep的所有hidden state
                                                          hidden_state0 in hidden_state_set if
                                                          V[timestep - 1][hidden_state0] > 0)
                V[timestep][hidden_state] = max_prob  # 找到最大的之后，作为结果记录在viterbi矩阵中
                new_path[hidden_state] = path[arg_max_prob_hidden_state] + [
                    hidden_state]  # 把新节点（hidden_state）添加到到达这个路径最大的那个hidden state对应的路径中
            path = new_path

        # 需要对最后一个timestep的节点单独处理
        if emission_pro['M'].get(observe_seq[-1], 0) > emission_pro['S'].get(observe_seq[-1], 0):
            prob, state = max([(V[-1][y], y) for y in ('E', 'M')])
        else:
            prob, state = max([(V[-1][y], y) for y in hidden_state_set])

        return prob, path[state]

    def cut(self):
        pass
