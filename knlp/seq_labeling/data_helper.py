# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: data_helper
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-20
# Description:
# -----------------------------------------------------------------------#
from knlp.utils.util import funtion_time_cost


class DataHelper:
    """
    这个类的主要作用是实现对序列标注的相关数据的预处理。
    因为原始数据往往是简单的使用空格或者什么标注得到的结果，并没有添加我们想要的标记
    所以需要额外的方法进行相关的处理
    """

    @classmethod
    @funtion_time_cost
    def make_smbe_data(cls, input_file, output_file):
        """
        这个函数将普通的以空格分割的标注数据变成带有smbe四个标记的训练数据

        Args:
            input_file: string
            output_file: string

        Returns: None

        """
        with open(input_file) as f:
            data = f.readlines()
        with open(output_file, "w") as fw:
            for line in data:
                word_list = line.strip().split()
                for word in word_list:
                    if len(word) == 1:
                        fw.write(word + "\tS\n")
                    else:
                        fw.write(word[0] + "\tB\n")
                        for char in word[1: len(word) - 1]:
                            fw.write(char + "\tM\n")
                        fw.write(word[-1] + "\tE\n")
                fw.write("\n")
        print("make smbe seg data done")

    @classmethod
    @funtion_time_cost
    def make_test_data(cls, input_file, output_file):
        """
        这个函数生成测试数据

        Args:
            input_file: string
            output_file: string

        Returns: None

        """
        with open(input_file) as f:
            data = f.readlines()
        with open(output_file, "w") as fw:
            for line in data:
                word_list = line.strip()
                for word in word_list:
                    # each word is a char in this situation
                    fw.write(word + "\tS\n")
                fw.write("\n")
        print("make smbe seg data done")

    @classmethod
    @funtion_time_cost
    def generate_vocab(cls, input_file, output_file):
        """
        这个函数生成vocab集合

        Args:
            input_file: string
            output_file: string

        Returns: None

        """
        with open(input_file) as f:
            data = f.readlines()
        vocab_set = set([])
        with open(output_file, "w") as fw:
            for line in data:
                word_list = line.strip()
                for word in word_list:
                    # each word is a char in this situation
                    vocab_set.add(word)
            for vocab in vocab_set:
                fw.write(vocab)
                fw.write("\n")
        print("make smbe seg data done")


if __name__ == '__main__':
    from knlp.common.constant import KNLP_PATH

    # make pku training data
    # input_file = KNLP_PATH + "/knlp/data/seg_data/icwb2-data/training/pku_training.utf8"
    # output_file = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_training_data.txt"
    # DataHelper.make_smbe_data(input_file, output_file)
    #
    # # make pku test data
    # input_file = KNLP_PATH + "/knlp/data/seg_data/icwb2-data/testing/pku_test.utf8"
    # output_file = KNLP_PATH + "/knlp/data/seg_data/train/pku_hmm_test_data.txt"
    # DataHelper.make_test_data(input_file, output_file)

    # make pku vocab data
    input_file = KNLP_PATH + "/knlp/data/seg_data/icwb2-data/testing/pku_test.utf8"
    output_file = KNLP_PATH + "/knlp/data/seg_data/train/pku_vocab.txt"
    DataHelper.generate_vocab(input_file, output_file)
