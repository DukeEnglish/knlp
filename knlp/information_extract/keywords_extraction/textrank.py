# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: textrank
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-04
# Description:
# -----------------------------------------------------------------------#


from knlp.common.constant import sentence_delimiters, allow_speech_tags
from knlp.information_extract.keywords_extraction.seg4IE import Segmentation
from knlp.utils.util import get_default_stop_words_file


class TextRank:
    """
    TextRank, base class for textrank keyword and textrank sentence
    ref https://github.com/letiantian/TextRank4ZH/blob/master/textrank4zh/
    """

    def __init__(self, stop_words_file=get_default_stop_words_file(), private_vocab=None,
                 allow_speech_tags=allow_speech_tags,
                 delimiters="|".join(sentence_delimiters)):
        """

        Args:
            stop_words_file: 停用词的文件路径
            private_vocab: list of string. 固定用来分词的。保证这些词会被分词生成。
            allow_speech_tags: 要保留的词性
            delimiters: 默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        """

        if not private_vocab:
            private_vocab = []

        # input
        self.text = ''

        # init seg
        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)

        self.sentences = None
        self.words_no_filter = None  # 对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words = None  # 去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters = None  # 保留words_no_stop_words中指定词性的单词而得到的两级列表。
        self.label_set = private_vocab
        self._vertex_source = None
        self._edge_source = None

    def analyze(self, text, lower=False, vertex_source='all_filters', edge_source='no_stop_words'):
        """
        分析文本

        Args:
            text: 文本内容，字符串。
            lower: 是否将文本转换为小写。默认为False。
            vertex_source: 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
            edge_source: 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。

        Returns:

        """

        self.text = text

        # 不同类型的分词结果以及分句结果
        seg_result = self.seg.segment(text=text, lower=lower)

        self.sentences = seg_result.sentences
        self.words_no_filter = seg_result.words_no_filter
        self.words_no_stop_words = seg_result.words_no_stop_words
        self.words_all_filters = seg_result.words_all_filters

        options = ['no_filter', 'no_stop_words', 'all_filters']

        # 构图
        if vertex_source in options:
            self._vertex_source = seg_result['words_' + vertex_source]
        else:
            self._vertex_source = seg_result['words_all_filters']

        if edge_source in options:
            self._edge_source = seg_result['words_' + edge_source]
        else:
            self._edge_source = seg_result['words_no_stop_words']
