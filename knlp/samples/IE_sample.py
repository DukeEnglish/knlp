# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: IE_sample
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-05
# Description:
# -----------------------------------------------------------------------#


from knlp.information_extract.keywords_extraction import TextRank4Keyword, TextRank4Sentence


def get_keyword(text, window=5, num=20, word_min_len=2, need_key_phrase=True):
    """
    获取文本中的关键词，采用text rank的算法获取

    Args:
        text: string
        window: 计算的时候的窗口大小
        num: 获取几个关键词
        word_min_len: 最小的词的长度
        need_key_phrase: 是否需要获取关键短语

    Returns: dict {"key_words": [[word, weight, count]], "key_phrase": [string], }

    """
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True)

    output = {"key_words": [], "key_phrase": []}

    for item in tr4w.get_keywords(num=num, word_min_len=word_min_len, window=window):
        kw_count = tr4w.text.count(item.word)
        output["key_words"].append([item.word, item.weight, kw_count])

    if need_key_phrase:
        for phrase in tr4w.get_keyphrases(keywords_num=10, min_occur_num=2):
            output['key_phrase'].append(phrase)

    return output


def get_key_sentences(text):
    """
    获取文本中的关键句子
    Args:
        text: string

    Returns: list of string

    """
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, edge_source='all_filters')

    output = []
    for item in tr4s.get_key_sentences(num=3):
        output.append(item.sentence)
    return output


def get_key_sentences_by_keyword(text):
    """
    获取文本中的关键句子
    Args:
        text: string

    Returns: list of string

    """
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, edge_source='all_filters')

    output = []
    for item in tr4s.get_key_sentences_by_keywords(num=3):
        output.append(item.sentence)
    return output


if __name__ == '__main__':
    with open("knlp/data/pytest_data.txt") as f:
        text = f.read()
    print(get_key_sentences(text))
    print(get_key_sentences_by_keyword(text))
    print(get_keyword(text))
