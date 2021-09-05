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

from knlp.common.constant import allow_speech_tags
from knlp.seq_labeling.hmm.inference import Inference
from knlp.utils.util import get_default_stop_words_file


class Segmentor(object):
    """
    This class define different method to do seg, and also including some basic training method


    """

    def __init__(self, stop_words_file=get_default_stop_words_file(), allow_speech_tags=allow_speech_tags,
                 private_vocab=None):
        """
        init some necessary params for this class

        Args:
            stop_words_file: string, 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
            allow_speech_tags: list, 词性列表，用于过滤。只保留需要保留的词性
        """
        self.stop_words = set()
        self.default_speech_tag_filter = allow_speech_tags
        self.private_vacab = private_vocab
        if self.private_vacab:
            for word in self.private_vacab:
                jieba.add_word(word, freq=None, tag=None)

        with open(stop_words_file, 'r') as f:
            for word in f:
                self.stop_words.add(word.strip())

    @staticmethod
    def del_word(word):
        jieba.del_word(word)

    def segment(self, text, function_name="jieba_cut", lower=True, use_stop_words=False, use_speech_tags_filter=False):
        """
        对一段文本进行分词，返回list类型的分词结果

        Args:
            text: string, 输入文本
            lower: 是否将单词小写（针对英文）
            use_stop_words: 若为True，则利用停止词集合来过滤（去掉停止词）
            use_speech_tags_filter: 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。目前只支持jieba的词性标注

        Returns: list of string

        """
        seg_method = getattr(self, function_name, None)
        if not seg_method:
            # TODO raise an exception
            return None
        word_list = seg_method(text)

        if function_name == "jieba_cut":  # 目前只支持jieba的词性标注
            if use_speech_tags_filter:
                word_list = [w for w in word_list if w.flag in self.default_speech_tag_filter]
                # 去除特殊符号
                word_list = [w.word.strip() for w in word_list if w.flag != 'x']
            else:
                word_list = [w.word.strip() for w in word_list]

        word_list = [word for word in word_list if len(word) > 0]

        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list

    @classmethod
    def jieba_cut(cls, sentence):
        """
        return result cut by jieba

        Args:
            sentence: string

        Returns: list of string

        """
        return jieba.posseg.cut(sentence)

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
