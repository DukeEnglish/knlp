import json
import sys
from collections import defaultdict

from knlp.common.constant import KNLP_PATH


class Train:
    """
    这个类要实现对以下四个信息的获取：
    状态集合：把所有的拼音记录下来，隐藏状态也是我们自己定义的标签，为拼音对应的汉字
    转移概率集合：我们可以从训练数据中获取相关的信息，P(T_n|T_n-1)，那么只要做个相应的统计就可以实现这个需求。
    发射概率：这个要统计的就是给定标签下，各个不同的汉字的概率是多少
    初始状态集合：所有的字都会有一个P(X|我)，P(X|是)，P(X|一)，P(X|个)，P(X|大)，P(X|好)，P(X|人)

    这个信息是输入信息和上面四个不太一样
    观测序列：这个是我们在inference的时候会使用到的信息
    """

    def __init__(self, vocab_set_path=None, training_data_path=None, test_data_path=None):

        self._state_set = {}
        self._transition_pro = {}
        self._transition_pro_ = {'data': self._transition_pro}
        self._emission_pro = {}
        self._emission_pro_={'data':self._emission_pro , 'default':1e-200}
        self._init_state_set = {}
        self.vocab_set_path = ""
        self.training_data_path = ""
        self.vocab_data = []
        self.training_data = []
        if vocab_set_path and training_data_path:
            self.init_variable(vocab_set_path=vocab_set_path, training_data_path=training_data_path,
                               test_data_path=test_data_path)


    def init_variable(self, vocab_set_path=None, training_data_path=None, test_data_path=None):
        self.vocab_set_path = KNLP_PATH + "/knlp/data/seg_data/train/pin_hanzi.txt" if not vocab_set_path else vocab_set_path
        self.training_data_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data_sample.txt" if not training_data_path else training_data_path
        # self.test_data_path = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt" if not test_data_path else test_data_path
        with open(self.vocab_set_path,encoding='utf-8') as f:
            self.vocab_data = f.readlines()

        with open(self.training_data_path,encoding='utf-8') as f:
            self.training_data = f.readlines()

    @property
    def state_set(self):
        self.set_state()
        return self._state_set

    @property
    def transition_pro(self):
        self.set_transition_pro()
        return self._transition_pro_

    @property
    def emission_pro(self):
        self.set_emission_pro()
        return self._emission_pro_

    @property
    def init_state_set(self):
        self.set_init_state_set()
        return self._init_state_set

    def set_state(self):
        self._state_set["hidden_state"] = []
        file = open(KNLP_PATH + "/knlp/data/seg_data/train/tag.txt", encoding='utf-8')
        for line in file.readlines():
            if not (line in self._state_set["hidden_state"] or line == '\n'):
                self._state_set["hidden_state"].append(line[0])


        self._state_set["observation_state"] = []
        file = open(KNLP_PATH + "/knlp/data/seg_data/train/out3.txt", encoding='utf-8')
        for line in file.readlines():
            if not (line in self._state_set["observation_state"] or line == '\n'):
                self._state_set["observation_state"].append(line[:-1])


    def set_transition_pro(self):
        """
        统计获取转移概率

        Returns:

        """


        count_dict = {}
        file = open(KNLP_PATH + "/knlp/data/seg_data/train/tag.txt", encoding='utf-8')
        for line in file.readlines():
            count_dict[line[0]] = defaultdict(int)


        for idx, line in enumerate(self.training_data):
            line = line.strip()
            total_lines = len(self.training_data)
            if not line:
                continue
            if (idx + 1) < total_lines:
                next_line = self.training_data[idx + 1].strip()  # 增加对访问下标超限的判断
            else:
                continue
            next_line = self.training_data[idx + 1].strip()
            if not next_line:
                continue
            next_line = self.training_data[idx + 1].strip().split(" ")  # 获取下一个标签
            count_dict[line[-1]][next_line[-1]] += 1
        for start_label, end_labels in count_dict.items():

            min = 10
            self._transition_pro[start_label] = {}
            cnt_sum = sum(list(end_labels.values()))
            for end_label, count in end_labels.items():
                self._transition_pro[start_label][end_label] = count / cnt_sum
                if (count / cnt_sum) < min:
                    min = count / cnt_sum

            tot1 = len(self._transition_pro[start_label])
            if tot1 != 0:
                self._transition_pro[start_label]['default'] = min / 10

        tot2 = len(self._transition_pro)
        if tot2 != 0:
            self._transition_pro['default'] = 1e-200


    def set_emission_pro(self):
        """
        统计获取发射概率

        Returns:

        """
        count_dict = {}
        file = open(KNLP_PATH + "/knlp/data/seg_data/train/tag.txt", encoding='utf-8')
        for line in file.readlines():
            count_dict[line[0]] = defaultdict(int)

        for line in self.training_data:
            if not line.strip():
                continue
            line = line.strip().split(" ")
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
        count_dict = {}
        file = open(KNLP_PATH + "/knlp/data/seg_data/train/tag.txt", encoding='utf-8')
        for line in file.readlines():
            count_dict[line[0]] = 0

        for line in self.training_data:
            if not line.strip():
                continue
            line = line.strip().split(" ")
            count_dict[line[-1]] += 1
        cnt_sum = sum(list(count_dict.values()))
        for start_label, cnt in count_dict.items():
            self._init_state_set[start_label] = cnt / cnt_sum

    def get_pinyin_dict(self):
        f = open(KNLP_PATH + "/knlp/data/seg_data/train/pin_hanzi.txt", encoding='utf-8')
        # 按行读取
        self.pinyin_to_chinese = {}
        for line in f.readlines():
            # 将每行拼音与汉字之间的零宽不换行空格换为普通空格

            # 将每行按空格切分并放入line列表中，一共有两个部分，其中line[0]为拼音，line[1]为对应的汉字（一堆）
            line = line.strip().split()
            # 存入拼音字典中
            if line[0] in self.pinyin_to_chinese.keys():
                if line[-1] not in self.pinyin_to_chinese[line[0]]:
                    self.pinyin_to_chinese[line[0]] = self.pinyin_to_chinese[line[0]] + line[-1]
            else:
                self.pinyin_to_chinese[line[0]] = line[-1]


    @staticmethod
    def save_model(file_path, data, format="json"):
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def build_model(self, state_set_save_path=None, transition_pro_save_path=None, emission_pro_save_path=None,
                    init_state_set_save_path=None):
        """
        依次运行以上的几个函数，然后将获取到的结果保存下来

        Returns:
        """
        state_set = KNLP_PATH + "/knlp/model/hmm/pin_state_set.json" if not state_set_save_path else state_set_save_path + "/pin_state_set.json"
        transition_pro = KNLP_PATH + "/knlp/model/hmm/pin_transition_pro.json" if not transition_pro_save_path else transition_pro_save_path + "/pin_transition_pro.json"
        emission_pro = KNLP_PATH + "/knlp/model/hmm/pin_emission_pro.json" if not emission_pro_save_path else emission_pro_save_path + "/pin_emission_pro.json"
        init_state_set = KNLP_PATH + "/knlp/model/hmm/pin_init_state_set.json" if not init_state_set_save_path else init_state_set_save_path + "/pin_init_state_set.json"
        pin_dic = KNLP_PATH + "/knlp/model/hmm/pinyin_dic.json" if not init_state_set_save_path else init_state_set_save_path + "/pinyin_dic.json"
        self.save_model(file_path=state_set, data=self.state_set)
        self.save_model(file_path=transition_pro, data=self.transition_pro)
        self.save_model(file_path=emission_pro, data=self.emission_pro)
        self.save_model(file_path=init_state_set, data=self.init_state_set)
        self.save_model(file_path=pin_dic, data=self.pin_dic)


if __name__ == '__main__':
    # input path for vacab and training data
    args = sys.argv
    vocab_set_path = None
    training_data_path = None
    print('训练中...')

    if len(args) > 1:
        vocab_set_path = args[1]
        training_data_path = args[2]

    a = Train(vocab_set_path=vocab_set_path, training_data_path=training_data_path)
    a.init_variable()
    a.build_model()
    print('训练完成')