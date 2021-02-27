#!/usr/bin/python
#-*- coding:UTF-8 -*-
#-----------------------------------------------------------------------#
# File Name: sentiment
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description: 
#-----------------------------------------------------------------------#

from snownlp import SnowNLP


class SentimentAnalysis:
    """
    This class could offer some functions to do sentiment analysis

    """
    def __init__(self):
        pass

    @classmethod
    def sentiment_snownlp(cls, sentence):
        """
        call snownlp to do sentiment analysis

        Args:
            sentence: string

        Returns: float

        """
        snow = SnowNLP(sentence)
        return snow.sentiments
