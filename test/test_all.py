#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: test
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
# -----------------------------------------------------------------------#

from cnlp import Cnlp


def test_all():
    with open("cnlp/data/pytest_data.txt") as f:
        text = f.read()
    res = Cnlp(text)
    print("seg_result is", res.seg_result)
    print("ner_result is", res.ner_result)
    print("sentiment score is", res.sentiment)
    print("key_words are", res.key_words)
    print("key sentences are", res.key_sentences)


if __name__ == '__main__':
    test_all()
