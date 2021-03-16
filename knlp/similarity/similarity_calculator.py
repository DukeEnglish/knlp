#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: similarity_calculator
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description:
# -----------------------------------------------------------------------#

from snownlp import SnowNLP


class SimilarityCalculator:
    """
    This class could offer some functions to do similarity calculator

    """

    def __init__(self):
        pass

    @classmethod
    def similarity_snownlp(cls, sentence1, sentence2):
        """
        call snownlp to do similarity analysis

        Args:
            sentence1: string
            sentence2: string

        Returns: float

        """
        snow = SnowNLP(sentence1)
        return snow.sim(sentence2)
