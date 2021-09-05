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
import re

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

    """

    def __init__(self):
        self._state_set = {}
        self._transition_pro = {}
        self._emission_pro = {}
        self._init_state_set = {}
        self._hidden_state_set = {}
        self.load_mode()
        for hidden_state in self._hidden_state_set:
            self.min_emission_pro = min([value for _, value in self._emission_pro[hidden_state].items()]) / 2

    def load_mode(self, state_set_save_path=None, transition_pro_save_path=None, emission_pro_save_path=None,
                  init_state_set_save_path=None, save_format="json"):
        def helper(file_path, save_format="json"):
            if save_format == "json":
                with open(file_path) as f:
                    return json.load(f)

        state_set = KNLP_PATH + "/knlp/model/hmm/state_set.json" if not state_set_save_path else state_set_save_path + "/state_set.json"
        transition_pro = KNLP_PATH + "/knlp/model/hmm/transition_pro.json" if not transition_pro_save_path else transition_pro_save_path + "/transition_pro.json"
        emission_pro = KNLP_PATH + "/knlp/model/hmm/emission_pro.json" if not emission_pro_save_path else emission_pro_save_path + "/emission_pro.json"
        init_state_set = KNLP_PATH + "/knlp/model/hmm/init_state_set.json" if not init_state_set_save_path else init_state_set_save_path + "/init_state_set.json"
        self._state_set = helper(file_path=state_set)
        self._hidden_state_set = self._state_set["hidden_state"]
        self._transition_pro = helper(file_path=transition_pro)
        self._emission_pro = helper(file_path=emission_pro)
        self._init_state_set = helper(file_path=init_state_set)

    def viterbi(self, observe_seq, hidden_state_set=None, init_state_set=None, transition_pro=None, emission_pro=None):
        if not hidden_state_set:
            hidden_state_set = self._hidden_state_set
        if not init_state_set:
            init_state_set = self._init_state_set
        if not transition_pro:
            transition_pro = self._transition_pro
        if not emission_pro:
            emission_pro = self._emission_pro
        viterbi_matrix = [{}]  # 每个timestep的几个概率大小，数组的index为timestep，里面的字典为概率值。可以想象为一个矩阵，横轴为timestep，纵轴为不同的概率值
        path = {}  # key 是当前的使整体概率最大的hidden state，value是一个数组，保存了路由到当前这个hidden state的，之前的所有的hidden state
        # 计算初始状态的概率分布，以及对应的路径， timestep = 1
        for hidden_state in hidden_state_set:
            viterbi_matrix[0][hidden_state] = init_state_set[hidden_state] * emission_pro[hidden_state].get(
                observe_seq[0], 0)
            path[hidden_state] = [hidden_state]
        # 计算后续几个状态的分布以及对应的路径，timestep > 1
        for timestep in range(1, len(observe_seq)):
            viterbi_matrix.append({})
            new_path = {}
            for hidden_state in hidden_state_set:  # 循环遍历这个timestep可以采用的每一个hidden state
                # 这里这个就是我们求解viterbi算法需要使用的公式
                # 这里做的就是将上一个timestep的每个状态到达当前的状态的概率大小计算了一下，得到最大的那个状态以及对应的概率值
                # 注意max在对数组处理的时候会用数组的第一个元素进行处理，如果第一个值一样，会用第二个进行
                max_prob, arg_max_prob_hidden_state = max([(
                    viterbi_matrix[timestep - 1][hidden_state0] * transition_pro[hidden_state0].get(hidden_state,
                                                                                                    0) *  # noqa
                    emission_pro[hidden_state].get(observe_seq[timestep], self.min_emission_pro), hidden_state0) for
                    hidden_state0 in hidden_state_set if
                    viterbi_matrix[timestep - 1][
                        hidden_state0] > 0])  # 这个for循环 循环的是前一个timestep的所有hidden state. 使用min score 来应对未登录字

                viterbi_matrix[timestep][hidden_state] = max_prob  # 找到最大的之后，作为结果记录在viterbi矩阵中
                # 把新节点（hidden_state）添加到到达这个路径最大的那个hidden state对应的路径中
                new_path[hidden_state] = path[arg_max_prob_hidden_state] + [hidden_state]
            path = new_path

        # 需要对最后一个timestep的节点单独处理
        prob, state = max([(viterbi_matrix[-1][y], y) for y in ("E", "S")])

        return prob, path[state]

    def __cut(self, sentence):
        # call viterbi to cut this sentence
        prob, pos_list = self.viterbi(sentence, ['B', 'M', 'E', 'S'])
        begin, next = 0, 0
        for idx, char in enumerate(sentence):
            pos = pos_list[idx]
            if pos == 'B':
                begin = idx
            elif pos == 'E':
                yield sentence[begin: idx + 1]
                next = idx + 1
            elif pos == 'S':
                yield char
                next = idx + 1
        if next < len(sentence):
            yield sentence[next:]

    def cut(self, sentence):
        re_zh, re_no_zh = re.compile("([\u4E00-\u9FA5]+)"), re.compile("[^a-zA-Z0-9+#\n]")  # 只对汉字做分词
        processed_sentence = re_zh.split(sentence)  # 按照汉字团进行分割
        for block in processed_sentence:
            if re_zh.match(block):  # 对汉字进行分词
                for word in self.__cut(block):
                    yield word
            else:
                for char in re_no_zh.split(block):  # 把剩下的分出来
                    if not char:
                        continue
                    yield char


if __name__ == '__main__':
    test = Inference()
    test_sen = "姚晨和老凌离婚了"
    test_sen = "本pkg的主体架构参考了snownlp和textblob，因为这种实现方式对于调用方来说最方便。"
    print((test.cut(test_sen)))
    print(list(test.cut(test_sen)))
