#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: __init__.py
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
# -----------------------------------------------------------------------#
from knlp.seq_labeling import seg, ner, evaluation_seg_files, evaluation_seg
from knlp.seq_sentiment import sentiment
from knlp.information_extract import get_keyword, get_key_sentences


class Knlp(object):
    """
    This class is designed for all the basic caller


    """

    def __init__(self, data=''):
        self.data = data

    @property
    def seg_result(self, function_name="jieba_cut"):
        """
        This function return the result cut by seg defined in seq_labeling

        Args:
            function_name: string

        Returns: list of words

        """
        return seg(self.data, function_name=function_name)

    @property
    def ner_result(self, function_name="jieba_ner"):
        """


        Args:
            function_name:string

        Returns: list of (word, flag) pair

        """
        return ner(sentence=self.data, function_name=function_name)

    @property
    def sentiment(self, function_name="sentiment_snownlp"):
        """
        This function could call different function to do sentiment analysis

        Args:
            function_name: string

        Returns:

        """
        return sentiment(self.data, function_name=function_name)

    @property
    def key_words(self, window=5, num=20, word_min_len=2, need_key_phrase=True):
        """
        获取文本中的关键词，采用text rank的算法获取
        Args:
            window: 计算的时候的窗口大小
            num: 获取几个关键词
            word_min_len: 最小的词的长度
            need_key_phrase: 是否需要获取关键短语

        Returns: dict {"key_words": [[word, weight, count]], "key_phrase": [string], }
        """
        return get_keyword(self.data, window=window, num=num, word_min_len=word_min_len,
                           need_key_phrase=need_key_phrase)

    @property
    def key_sentences(self):
        """
        获取文本中的关键句子

        Returns: list of string, string is key sentence
        """
        return get_key_sentences(self.data)

    @staticmethod
    def evaluation_segment(seg_result_gold, seg_result_pred, seg_symbol=" "):
        """

        Args:
            seg_result_gold: string, seg result separated by seg_symbol, gold_result
            seg_result_pred: string, seg result separated by seg_symbol, predicted_result
            seg_symbol: string,

        Returns: precision, recall, f1, all of them are float

        """
        return evaluation_seg(seg_result_gold, seg_result_pred, seg_symbol)

    @staticmethod
    def evaluation_segment_file(gold_file_name, pred_file_name, seg_symbol=" "):
        """

        Args:
            gold_file_name: string, seg result separated by seg_symbol, gold_file_name
            pred_file_name: string, seg result separated by seg_symbol, pred_file_name
            seg_symbol: string,

        Returns: precision, recall, f1, all of them are float

        """
        return evaluation_seg_files(gold_file_name, pred_file_name, seg_symbol)
    # @property
    # def sentences(self):
    #     return normal.get_sentences(self.doc)
    #
    # @property
    # def han(self):
    #     return normal.zh2hans(self.doc)
    #
    # @property
    # def pinyin(self):
    #     return normal.get_pinyin(self.doc)
    #
    # @property
    # def tf(self):
    #     return self.bm25.f
    #
    # @property
    # def idf(self):
    #     return self.bm25.idf

    # def sim(self, doc):
    #     return self.bm25.simall(doc)
    #
    # def summary(self, limit=5):
    #     doc = []
    #     sents = self.sentences
    #     for sent in sents:
    #         words = seg.seg(sent)
    #         words = normal.filter_stop(words)
    #         doc.append(words)
    #     rank = textrank.TextRank(doc)
    #     rank.solve()
    #     ret = []
    #     for index in rank.top_index(limit):
    #         ret.append(sents[index])
    #     return ret
    #
    # def keywords(self, limit=5, merge=False):
    #     doc = []
    #     sents = self.sentences
    #     for sent in sents:
    #         words = seg.seg(sent)
    #         words = normal.filter_stop(words)
    #         doc.append(words)
    #     rank = textrank.KeywordTextRank(doc)
    #     rank.solve()
    #     ret = []
    #     for w in rank.top_index(limit):
    #         ret.append(w)
    #     if merge:
    #         wm = words_merge.SimpleMerge(self.doc, ret)
    #         return wm.merge()
    #     return ret
