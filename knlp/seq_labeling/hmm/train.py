# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-18
# Description:
# -----------------------------------------------------------------------#
"""
针对hmm的训练和推理，要清晰的分为不同的步骤，分步骤进行实现
0 清晰定义好hmm需要的几个参数
1 完成基于训练数据的hmm相关参数的代码并且完成相关的编码实现
2 完成hmm的inference相关的代码

针对序列标注问题，什么是可见状态，什么是隐藏状态，转移概率，发射概率
分词，我们这样应对：
SBME
我 是 一个 大 好 人
S  S  BE  B M E
可见状态：几个汉字
隐藏状态：几个字母，标注结果
转移概率：就是隐藏状态互相之间的概率大小
发射概率：是从隐藏状态到可见状态概率大小
初始状态：

换句话说，在inference的时候：
输入是
我 是 一个 大 好 人
输出则是对应的每个字下的标签是什么
直观感受来说，我们可以假设，不同的汉字对应于不同的标签有不同的概率大小，这个可以通过统计得到，经过大量的统计之后，我们可以知道给定一个汉字，它所对应各种标记的概率大小
则此时我们可以知道P(X|我)，P(X|是)，P(X|一)，P(X|个)，P(X|大)，P(X|好)，P(X|人)的大小分别是多少，例如：
P(X|我)就有四个结果：(随便举例的)
P(S|我) = 0.4
P(M|我) = 0.2
P(B|我) = 0.3
P(E|我) = 0.1
最后我们可以将他们乘起来【P(X|我)，P(X|是)，P(X|一)，P(X|个)，P(X|大)，P(X|好)，P(X|人)】，得到一整句话下使用不同标记的概率大小，找到概率最大的那个即可。
这就是最直觉的基于统计的分词方法。
可是，我们后知后觉的想一想，是不是前一个字的标签对后一个字的标签应该是有一定暗示意义的。比如，如果前一个字是S那么后一个字就不应该出现E和M。
这就涉及到前一个状态对后一个状态的影响，所以我们提出一个转移概率，就有马尔可夫链，而对于一阶马尔可夫，就是假设每一个标签只和前一个标签有关。
Q：一个状态集合
A：转移概率几何
init_P：初始概率。很直观的理解就是第一个字是什么标签，是会有一个统计出来的概率值的。

可是，这个时候我们就发现，我们其实是不知道一个字背后的标签的，所以我们需要把这两个信息合起来考虑。
所以HMM就允许我们同时谈论观测事件（我们看到的words）和隐藏事件（例如pos tag），
我们考虑让他们作为我们概率模型的因果变量。一个HMM模型可以被定义如下：
Q：一个状态集合：隐藏状态和观测状态
A：转移概率集合：隐藏状态之间的转移概率
O：观测序列：输入的序列
B：发射概率，表示从隐藏状态到观测状态的概率
初始状态集合：很直观的理解就是第一个字是什么标签，是会有一个统计出来的概率值的。

以上的这几种信息都是可以从给定的训练数据集中获取到的。
一个个看一下：
状态集合：把所有的汉字记录下来，隐藏状态也是我们自己定义的标签
转移概率集合：我们可以从训练数据中获取相关的信息，P(T_n|T_n-1)，那么只要做个相应的统计就可以实现这个需求。
观测序列：这个是我们在inference的时候会使用到的信息
发射概率：这个要统计的就是给定标签下，各个不同的汉字的概率是多少
初始状态集合：所有的标签开头的概率大小
从以上的分析看来，一个分词要做的就是对一对数据的统计代码。

好，最后我们得到的一个模型，存储了以上的信息
然后我们完成一个inference的代码，利用这个模型进行对新输入的观测序列的inference
"""
import json
import sys
from collections import defaultdict

from knlp.common.constant import KNLP_PATH


class Train:
    """
    这个类要实现对以下四个信息的获取：
    状态集合：把所有的汉字记录下来，隐藏状态也是我们自己定义的标签
    转移概率集合：我们可以从训练数据中获取相关的信息，P(T_n|T_n-1)，那么只要做个相应的统计就可以实现这个需求。
    发射概率：这个要统计的就是给定标签下，各个不同的汉字的概率是多少
    初始状态集合：所有的字都会有一个P(X|我)，P(X|是)，P(X|一)，P(X|个)，P(X|大)，P(X|好)，P(X|人)
    从以上的分析看来，一个分词要做的就是对一对数据的统计代码。

    这个信息是输入信息和上面四个不太一样
    观测序列：这个是我们在inference的时候会使用到的信息
    """

    def __init__(self, vocab_set_path=None, training_data_path=None, test_data_path=None):

        self._state_set = {}
        self._transition_pro = {}
        self._emission_pro = {}
        self._init_state_set = {}
        self.vocab_set_path = ""
        self.training_data_path = ""
        self.vocab_data = []
        self.training_data = []
        if vocab_set_path and training_data_path:
            self.init_variable(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                               test_data_path=test_data_path)

    def init_variable(self, vocab_set_path=None, training_data_path=None, test_data_path=None):
        self.vocab_set_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt" if not vocab_set_path else vocab_set_path
        self.training_data_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt" if not training_data_path else training_data_path
        # self.test_data_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt" if not test_data_path else test_data_path
        with open(self.vocab_set_path) as f:
            self.vocab_data = f.readlines()

        with open(self.training_data_path) as f:
            self.training_data = f.readlines()

    @property
    def state_set(self):
        self.set_state()
        return self._state_set

    @property
    def transition_pro(self):
        self.set_transition_pro()
        return self._transition_pro

    @property
    def emission_pro(self):
        self.set_emission_pro()
        return self._emission_pro

    @property
    def init_state_set(self):
        self.set_init_state_set()
        return self._init_state_set

    def set_state(self):
        self._state_set["hidden_state"] = ["S", "B", "E", "M"]
        self._state_set["observation_state"] = []
        for line in self.vocab_data:
            self._state_set["observation_state"].append(line.strip())

    def set_transition_pro(self):
        """
        统计获取转移概率
        S: B E M
        Returns:

        """

        count_dict = {
            "S": defaultdict(int),
            "B": defaultdict(int),
            "M": defaultdict(int),
            "E": defaultdict(int),
        }
        for idx, line in enumerate(self.training_data):
            line = line.strip()
            if not line:
                continue
            line = line.strip().split("\t")  # 获取到当前正在统计的那个标签
            next_line = self.training_data[idx + 1].strip()
            if not next_line:
                continue
            next_line = self.training_data[idx + 1].strip().split("\t")  # 获取下一个标签
            count_dict[line[-1]][next_line[-1]] += 1
        for start_label, end_labels in count_dict.items():

            self._transition_pro[start_label] = {}
            cnt_sum = sum(list(end_labels.values()))
            for end_label, count in end_labels.items():
                self._transition_pro[start_label][end_label] = count / cnt_sum

    def set_emission_pro(self):
        """
        统计获取发射概率

        Returns:

        """
        count_dict = {
            "S": defaultdict(int),
            "B": defaultdict(int),
            "M": defaultdict(int),
            "E": defaultdict(int),
        }
        for line in self.training_data:
            if not line.strip():
                continue
            line = line.strip().split("\t")
            count_dict[line[-1]][line[0]] += 1
        for hidden_state, observation_states in count_dict.items():
            self._emission_pro[hidden_state] = {}
            cnt_sum = sum(list(observation_states.values()))
            for observation_state, count in observation_states.items():
                self._emission_pro[hidden_state][observation_state] = count / cnt_sum

    def set_init_state_set(self):
        """

        当这个字开头的时候，有多大的概率是哪个标签
        {WORD: {LABEL: PRO}}
        Returns:

        """
        count_dict = {
            "S": 0,
            "B": 0,
            "M": 0,
            "E": 0,
        }
        for line in self.training_data:
            if not line.strip():
                continue
            line = line.strip().split("\t")
            count_dict[line[-1]] += 1
        cnt_sum = sum(list(count_dict.values()))
        for start_label, cnt in count_dict.items():
            self._init_state_set[start_label] = cnt / cnt_sum

    @staticmethod
    def save_model(file_path, data, format="json"):
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(data, f, ensure_ascii=False)

    def build_model(self, state_set_save_path=None, transition_pro_save_path=None, emission_pro_save_path=None,
                    init_state_set_save_path=None):
        """
        依次运行以上的几个函数，然后将获取到的结果保存下来

        Returns:
        """
        state_set = KNLP_PATH + "/knlp/model/hmm/state_set.json" if not state_set_save_path else state_set_save_path + "/state_set.json"
        transition_pro = KNLP_PATH + "/knlp/model/hmm/transition_pro.json" if not transition_pro_save_path else transition_pro_save_path + "/transition_pro.json"
        emission_pro = KNLP_PATH + "/knlp/model/hmm/emission_pro.json" if not emission_pro_save_path else emission_pro_save_path + "/emission_pro.json"
        init_state_set = KNLP_PATH + "/knlp/model/hmm/init_state_set.json" if not init_state_set_save_path else init_state_set_save_path + "/init_state_set.json"
        self.save_model(file_path=state_set, data=self.state_set)
        self.save_model(file_path=transition_pro, data=self.transition_pro)
        self.save_model(file_path=emission_pro, data=self.emission_pro)
        self.save_model(file_path=init_state_set, data=self.init_state_set)


if __name__ == '__main__':
    # input path for vacab and training data
    args = sys.argv
    vocab_set_path = None
    training_data_path = None

    if len(args) > 1:
        vocab_set_path = args[1]
        training_data_path = args[2]

    a = Train(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
    a.init_variable()
    a.build_model()
