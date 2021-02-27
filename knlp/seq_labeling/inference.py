#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
# -----------------------------------------------------------------------#

from knlp.seq_labeling.seg import Segmentor
from knlp.seq_labeling.ner import NER


def seg(sentence, function_name="jieba_cut"):
    """
        This function could call different function to cut sentence

    Args:
        sentence: string
        function_name: string

    Returns: list of word

    """
    words = []
    seg_method = getattr(Segmentor, function_name, None)
    if not seg_method:
        # TODO raise an exception
        return None
    for word in seg_method(sentence):
        word = word.strip()
        if not word:
            continue
        words.append(word)
    return words


def ner(sentence, function_name="jieba_ner"):
    """
    This function could return the ner res of sentence via different function

    Args:
        sentence: string
        function_name: string

    Returns: list of pairs (word, tag)

    """
    word_tags = []
    ner_method = getattr(NER, function_name, None)
    if not ner_method:
        # TODO raise an exception
        return None
    for word_tag in ner_method(sentence):
        if not word_tag:
            continue
        word_tags.append(word_tag)
    return word_tags
