# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: data_helper
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-20
# Description:
# -----------------------------------------------------------------------#
from knlp.utils.util import

class DataHelper:
    """
    这个类的主要作用是实现对序列标注的相关数据的预处理。
    因为原始数据往往是简单的使用空格或者什么标注得到的结果，并没有添加我们想要的标记
    所以需要额外的方法进行相关的处理
    """

    @
    @classmethod
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
                word = line.strip().split(" ")
                if len(word) == 1:
                    fw.write(word[0] + "\tS\n")
                else:
                    fw.write(word[0] + "\tB\n")
                    for char in word[1: len(word) - 1]:
                        fw.write(char + "\tM\n")
                    fw.write(word[-1] + "\tE\n")
        print("make smbe seg data done")


if __name__ == '__main__':
    input_file = ""
    output_file = ""
    DataHelper.make_smbe_data(input_file, output_file)

