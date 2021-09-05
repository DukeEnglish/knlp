# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: seg_sample
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-05
# Description:
# -----------------------------------------------------------------------#
from knlp.seq_labeling import Segmentor


def sample_seg(text):
    segmentor_before = Segmentor()
    res_before = segmentor_before.segment(text=text)
    segmentor_after = Segmentor(private_vocab=["固定词语", "搭配"])
    res_after = segmentor_after.segment(text=text)
    return res_before, res_after


if __name__ == '__main__':
    text = "固定词语的搭配测试"
    res = sample_seg(text)
    print(res)

