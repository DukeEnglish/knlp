#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: utils
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description: 
# -----------------------------------------------------------------------#


def word2pair(segment_result, seg_symbol=" "):
    """
    return idx pair for segment_result
    Args:
        segment_result: string
        seg_symbol: string

    Returns: [idx_start, idx_end]

    """
    line = segment_result.strip().split(seg_symbol)
    res = []
    idx = 0
    for word in line:
        res.append([idx, idx + len(word)])
        idx += len(word)
    return res


def evaluation_seg(seg_result_gold, seg_result_pred, seg_symbol=" "):
    """

    Args:
        seg_result_gold: string, seg result separated by seg_symbol, gold_result
        seg_result_pred: string, seg result separated by seg_symbol, predicted_result
        seg_symbol: string,

    Returns: precision, recall, f1, all of them are float

    """
    idx_gold = word2pair(seg_result_gold, seg_symbol)
    idx_pred = word2pair(seg_result_pred, seg_symbol)
    res = [x for x in idx_gold if x in idx_pred]
    precision = len(res) / len(idx_pred)
    recall = len(res) / len(idx_gold)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'precision: {precision}, recall: {recall}, f1: {f1}')
    return precision, recall, f1


def evaluation_seg_files(gold_file_name, pred_file_name, seg_symbol=" "):
    """

    Args:
        gold_file_name: string, seg result separated by seg_symbol, gold_file_name
        pred_file_name: string, seg result separated by seg_symbol, pred_file_name
        seg_symbol: string,

    Returns: precision, recall, f1, all of them are float

    """
    with open(gold_file_name) as g:
        gold_res_string = seg_symbol.join(g.readlines())
    with open(pred_file_name) as g:
        pred_res_string = seg_symbol.join(g.readlines())
    return evaluation_seg(gold_res_string, pred_res_string, seg_symbol)

# if __name__ == '__main__':
#     evaluation()
