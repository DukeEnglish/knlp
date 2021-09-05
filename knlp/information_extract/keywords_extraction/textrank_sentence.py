# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: textrank_keyword
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-04
# Description:
# -----------------------------------------------------------------------#
import math

import networkx as nx
import numpy as np

from knlp.common.constant import sentence_delimiters, allow_speech_tags
from knlp.information_extract.keywords_extraction.textrank_keyword import TextRank4Keyword
from knlp.utils.util import AttrDict, get_default_stop_words_file


class TextRank4Sentence(TextRank4Keyword):
    """
    这个函数实现了利用Text rank算法来实现关键词提取的功能。
    基础的思路就是先分词，然后计算每个词语的权重，最后按照顺序排列，得到重要性
    暂时不考虑英文的需求
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

    def get_key_sentences(self, num=6, sentence_min_len=6, page_rank_config={'alpha': 0.85, }):
        """
        获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。

        Args:
            num:
            sentence_min_len:
            page_rank_config:

        Returns: 多个句子组成的列表。

        """
        key_sentences = self.sort_sentences(sentences=self.sentences,
                                            words=self._edge_source,
                                            sim_func=None,
                                            page_rank_config=page_rank_config)
        result = []
        count = 0
        for item in key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result

    def sort_sentences(self, sentences, words, sim_func=None, page_rank_config=None):
        """
        将句子按照关键程度从大到小排序
        Args:
            sentences: list of sentences
            words: 二维列表，子列表和sentences中的句子对应，子列表由单词组成
            sim_func: 计算两个句子的相似性，参数是两个由单词组成的列表
            page_rank_config: pagerank的设置

        Returns:

        """

        # TODO 修改为词向量的cosin
        def get_similarity(word_list1, word_list2):
            """
            计算两个句子的相似度。利用分词后的结果，计算共现的词语个数。

            Args:
                word_list1: list of words
                word_list2:

            Returns:

            """
            words = list(set(word_list1 + word_list2))
            vector1 = [float(word_list1.count(word)) for word in words]
            vector2 = [float(word_list2.count(word)) for word in words]

            vector3 = [vector1[x] * vector2[x] for x in range(len(vector1))]
            vector4 = [1 for num in vector3 if num > 0.]
            co_occur_num = sum(vector4)

            if abs(co_occur_num) <= 1e-12:
                return 0.

            denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2)))  # 分母

            if abs(denominator) < 1e-12:
                return 0.

            return co_occur_num / denominator

        if not sim_func:
            sim_func = get_similarity

        page_rank_config = {'alpha': 0.85, } if not page_rank_config else page_rank_config
        sorted_sentences = []
        _source = words
        sentences_num = len(_source)
        graph = np.zeros((sentences_num, sentences_num))

        for x in range(sentences_num):
            for y in range(x, sentences_num):
                similarity = sim_func(_source[x], _source[y])
                graph[x, y] = similarity
                graph[y, x] = similarity

        nx_graph = nx.from_numpy_matrix(graph)
        scores = nx.pagerank(nx_graph, **page_rank_config)  # this is a dict
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        for index, score in sorted_scores:
            item = AttrDict(index=index, sentence=sentences[index], weight=score)
            sorted_sentences.append(item)

        return sorted_sentences

    def get_key_sentences_by_keywords(self, num=6, sentence_min_len=6, kw_num=6, window=2, word_min_len=1,
                                      page_rank_config={'alpha': 0.85, }):
        """
        获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。

        Args:
            num:
            sentence_min_len:
            page_rank_config:

        Returns: 多个句子组成的列表。

        """
        key_sentences = self.sort_sentence_by_keyword(num=kw_num, window=window, word_min_len=word_min_len,
                                                      page_rank_config=page_rank_config)

        result = []
        count = 0
        for item in key_sentences:
            if count >= num:
                break
            if len(item['sentence']) >= sentence_min_len:
                result.append(item)
                count += 1
        return result

    def sort_sentence_by_keyword(self, num=6, window=2, word_min_len=1, page_rank_config={'alpha': 0.85, }):
        """
        通过关键词的个数进行句子的关键程度排序

        Returns:

        """
        sorted_sentences = []
        key_words = self.get_keywords(num=num, window=window, word_min_len=word_min_len,
                                      page_rank_config=page_rank_config)
        word2weight = {}
        for w in key_words:
            word2weight[w.word] = w.weight
        sentence2weight = {}

        # 基类中已经将所有的句子都分割好了。
        sentence_list = self.words_no_filter  # list of word_list
        for sentence_word_list in sentence_list:
            weight_of_word = []
            for sentence_word in sentence_word_list:
                if sentence_word in word2weight:
                    weight_of_word.append(word2weight[sentence_word])
            sentence2weight["".join(sentence_word_list)] = sum(weight_of_word)

        sorted_scores = sorted(sentence2weight.items(), key=lambda item: item[1], reverse=True)

        for sentence, score in sorted_scores:
            item = AttrDict(sentence=sentence, weight=score)
            sorted_sentences.append(item)

        return sorted_sentences


if __name__ == '__main__':
    with open("knlp/data/pytest_data.txt") as f:
        text = f.read()

    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, edge_source='all_filters')

    output = []
    for item in tr4s.get_key_sentences(num=3):
        output.append(item.sentence)
    print(output)

    for item in tr4s.get_key_sentences_by_keywords(num=3):
        output.append(item.sentence)
    print(output)
