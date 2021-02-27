#!/usr/bin/python
#-*- coding:UTF-8 -*-
#-----------------------------------------------------------------------#
# File Name: seg
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
#-----------------------------------------------------------------------#

import jieba


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