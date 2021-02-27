#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: test
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
# -----------------------------------------------------------------------#

from knlp import Knlp


def test_all():
    with open("knlp/data/pytest_data.txt") as f:
        text = f.read()
    res = Knlp(text)
    print("seg_result is", res.seg_result)
    print("ner_result is", res.ner_result)
    print("sentiment score is", res.sentiment)
    print("key_words are", res.key_words)
    print("key sentences are", res.key_sentences)
    gt_string = '就读 于 中国人民大学 电视 上 的 电影 节目 项目 的 研究 角色 本人 将 会 参与 配音'
    pred_string = '就读 于 中国 人民 大学 电视 上 的 电影 节目 项 目的 研究 角色 本人 将 会 参与 配音'
    print("evaluation res are", res.evaluation_segment(gt_string, pred_string))


if __name__ == '__main__':
    test_all()
