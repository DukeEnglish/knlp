#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: Segmentation.py
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2019-07-03
# Description:
# -----------------------------------------------------------------------#

import re

from knlp.common.constant import allow_speech_tags, sentence_delimiters
from knlp.seq_labeling.seg import Segmentor
from knlp.utils.util import get_default_stop_words_file, AttrDict


class WordSegmentation(Segmentor):
    """ 封装的分词 for 关键词提取 """

    def segment_sentences(self, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False):
        """
        将列表sequences中的每个元素/句子转换为由单词构成的列表。

        Args:
            sentences: list of strings
            lower:
            use_stop_words:
            use_speech_tags_filter:

        Returns: list of list of words [[word]]

        """
        res = []

        for sentence in sentences:
            res.append(self.segment(text=sentence,
                                    lower=lower,
                                    use_stop_words=use_stop_words,
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res


class SentenceSegmentation(object):
    """ 分句 """

    def __init__(self, delimiters="|".join(sentence_delimiters)):
        """


        Args:
            delimiters: string, separator for sentence
        """
        self.delimiters = delimiters

    def segment(self, text):
        """


        Args:
            text: string, a passage or a sentence

        Returns: list of sentence

        """
        # text is a passage seq is a paragraph
        symbol = "[" + self.delimiters + "]+"
        res = re.split(symbol, text)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res


class Segmentation(object):

    def __init__(self, stop_words_file=get_default_stop_words_file(), allow_speech_tags=allow_speech_tags,
                 delimiters="|".join(sentence_delimiters), private_vocab=None):
        """

        Args:
            stop_words_file:
            allow_speech_tags:
            delimiters:
        """

        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags, private_vocab=private_vocab)
        self.ss = SentenceSegmentation(delimiters=delimiters)

    def segment(self, text, lower=False):
        """


        Args:
            text: string, a sentence or a passage
            lower:

        Returns: AttrDict(
            sentences=sentences,
            words_no_filter=words_no_filter,
            words_no_stop_words=words_no_stop_words,
            words_all_filters=words_all_filters
        )

        """
        sentences = self.ss.segment(text)
        words_no_filter = self.ws.segment_sentences(sentences=sentences,
                                                    lower=lower,
                                                    use_stop_words=False,
                                                    use_speech_tags_filter=False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences,
                                                        lower=lower,
                                                        use_stop_words=True,
                                                        use_speech_tags_filter=False)
        words_all_filters = self.ws.segment_sentences(sentences=sentences,
                                                      lower=lower,
                                                      use_stop_words=True,
                                                      use_speech_tags_filter=True)

        return AttrDict(
            sentences=sentences,
            words_no_filter=words_no_filter,
            words_no_stop_words=words_no_stop_words,
            words_all_filters=words_all_filters
        )


if __name__ == '__main__':
    text = "测试分词的结果是否符合预期"
    pass
