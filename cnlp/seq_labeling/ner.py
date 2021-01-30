#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: ner
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
# -----------------------------------------------------------------------#

import jieba.posseg as pseg


class NER(object):
    """
    This class define different method to do NER, and also including some basic training method


    """

    def __init__(self):
        pass

    @classmethod
    def jieba_ner(cls, sentence):
        """
        return result tagged by jieba

        Args:
            sentence: string

        Returns: list of (word, flag) pair

        """
        res = []
        words = pseg.cut(sentence)  # jieba默认模式
        for word, flag in words:
            print('%s %s' % (word, flag))
            res.append((word, flag))
        return res
