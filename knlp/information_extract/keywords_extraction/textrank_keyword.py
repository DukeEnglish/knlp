# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: textrank_keyword
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-04
# Description:
# -----------------------------------------------------------------------#


import networkx as nx
import numpy as np

from knlp.common.constant import sentence_delimiters, allow_speech_tags
from knlp.information_extract.keywords_extraction.textrank import TextRank
from knlp.utils.util import get_default_stop_words_file, AttrDict


class TextRank4Keyword(TextRank):
    """
    这个函数实现了利用Text rank算法来实现关键词提取的功能。
    基础的思路就是先分词，然后计算每个词语的权重，最后按照顺序排列，得到重要性
    暂时不考虑英文的需求
    介绍请见 https://www.jiqizhixin.com/articles/2018-12-28-18
    ref https://github.com/letiantian/TextRank4ZH/blob/master/textrank4zh/
    """

    def __init__(self, stop_words_file=get_default_stop_words_file(), private_vocab=None,
                 allow_speech_tags=allow_speech_tags,
                 delimiters="|".join(sentence_delimiters)):
        """

        Args:
            stop_words_file: 停用词的文件路径
            label_set:
            allow_speech_tags: 要保留的词性
            delimiters: 默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        """
        super().__init__(stop_words_file=stop_words_file, private_vocab=private_vocab,
                         allow_speech_tags=allow_speech_tags,
                         delimiters=delimiters)

    def get_keywords(self, num=6, window=2, word_min_len=1, page_rank_config={'alpha': 0.85, }):
        """
        获取最重要的num个长度大于等于word_min_len的关键词。

        Args:
            num:
            window:
            word_min_len:
            page_rank_config:

        Returns: 关键词列表。AttriDict {}

        """
        # 获取按照text rank的方式得到的关键词，并排序
        keywords = self.sort_words(self._vertex_source, self._edge_source, window=window,
                                   page_rank_config=page_rank_config)

        result = []
        count = 0
        for item in keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result

    def get_keyphrases(self, keywords_num=12, min_occur_num=2):
        """
        获取关键短语。
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。
        使用有限的keywords_num 个关键词来构造短语


        Args:
            keywords_num: 关键词的个数
            min_occur_num: 最少出现次数

        Returns: 关键短语的列表。

        """
        keywords_set = set([item.word for item in self.get_keywords(num=keywords_num, word_min_len=1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) > 1:
                        keyphrases.add(''.join(one))  # concat在一起
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) > 1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases
                if self.text.count(phrase) >= min_occur_num or phrase in self.label_set]

    @staticmethod
    def sort_words(vertex_source, edge_source, window=2, page_rank_config=None):
        """
        将单词按关键程度从大到小排序

        Args:
            vertex_source: 二维列表，子列表代表句子，子列表的元素是单词，这些单词用来构造pagerank中的节点
            edge_source: 二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
            window: 一个句子中相邻的window个单词，两两之间认为有边
            page_rank_config: pagerank的设置

        Returns:

        """
        page_rank_config = {'alpha': 0.85, } if not page_rank_config else page_rank_config
        sorted_words = []
        word_index = {}
        index_word = {}
        _vertex_source = vertex_source
        _edge_source = edge_source
        words_number = 0
        for word_list in _vertex_source:
            for word in word_list:
                if word not in word_index:
                    word_index[word] = words_number
                    index_word[words_number] = word
                    # MAP WORD TO AN INDEX
                    words_number += 1

        graph = np.zeros((words_number, words_number))  # words_number X words_number MATRIX

        def combine(word_list, window=2):
            """
            构造在window下的单词组合，用来构造单词之间的边。

            Args:
                word_list: list of str, 由单词组成的列表。
                window: int, 窗口大小。

            Returns:

            """
            if window < 2:
                window = 2
            for x in range(1, window):
                if x >= len(word_list):
                    break
                word_list2 = word_list[x:]
                res = zip(word_list, word_list2)
                for r in res:
                    yield r

        for word_list in _edge_source:
            for w1, w2 in combine(word_list, window):
                if w1 in word_index and w2 in word_index:
                    index1 = word_index[w1]
                    index2 = word_index[w2]
                    #  有链接的位置 = 1。0
                    graph[index1][index2] = 1.0
                    graph[index2][index1] = 1.0

        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **page_rank_config)  # this is a dict DIRECTLY GET THE SCORE FOR ALL THIS MATRIX
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for index, score in sorted_scores:
            item = AttrDict(word=index_word[index], weight=score)
            sorted_words.append(item)

        return sorted_words


if __name__ == '__main__':
    text = "测试分词的结果是否符合预期"
    window = 5
    num = 20
    word_min_len = 2
    need_key_phrase = True
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True)

    output = {"key_words": [], "key_phrase": []}

    res_keywords = tr4w.get_keywords(num=num, word_min_len=word_min_len, window=window)

    for item in res_keywords:
        kw_count = tr4w.text.count(item.word)
        output["key_words"].append([item.word, item.weight, kw_count])

    if need_key_phrase:
        for phrase in tr4w.get_keyphrases(keywords_num=10, min_occur_num=2):
            output['key_phrase'].append(phrase)

    print(output)
