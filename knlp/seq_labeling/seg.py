#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: seg
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description:
# -----------------------------------------------------------------------#

import jieba

from knlp.seq_labeling.hmm.inference import Inference


class Segmentor(object):
    """
    This class define different method to do seg, and also including some basic training method


    """

    def __init__(self):
        pass

    @classmethod
    def jieba_cut(cls, sentence):
        """
        return result cut by jieba

        Args:
            sentence: string

        Returns: list of string

        """
        return jieba.lcut(sentence)

    @classmethod
    def hmm_seg(cls, sentence, model=None):
        """
        return result cut by hmm

        Args:
            sentence: string
            model:

        Returns: list of string

        """
        test = Inference()
        return list(test.cut(sentence))

    @classmethod
    def crf_seg(cls, sentence, model):
        """
        return result cut by crf

        Args:
            sentence: string
            model:

        Returns: list of string

        """
        pass

    @classmethod
    def trie_seg(cls, sentence, model):
        """
        return result cut by trie

        Args:
            sentence: string
            model:

        Returns: list of string

        """
        pass
