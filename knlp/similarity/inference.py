#!/usr/bin/python
#-*- coding:UTF-8 -*-
#-----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description: 
#-----------------------------------------------------------------------#

from knlp.similarity.similarity_calculator import SimilarityCalculator


def sentiment(sentence1, sentence2, function_name="similarity_snownlp"):
    """
        This function could call different function to do similarity analysis

    Args:
        sentence1: string
        sentence2: string
        function_name: string

    Returns: list of word

    """
    similarity_calculator = getattr(SimilarityCalculator, function_name, None)
    if not similarity_calculator:
        # TODO raise an exception
        return None
    return similarity_calculator(sentence1, sentence2)
