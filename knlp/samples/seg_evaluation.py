# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: seg_evaluation
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-05
# Description:
# -----------------------------------------------------------------------#


from knlp.seq_labeling import evaluation_seg


def sample_seg_evaluation(seg_result_gold, seg_result_pred, seg_symbol=" "):
    """

    Args:
        seg_result_gold: string, seg result separated by seg_symbol, gold_result
        seg_result_pred: string, seg result separated by seg_symbol, predicted_result
        seg_symbol: string, separator for seg result

    Returns: precision, recall, f1, all of them are float

    """
    return evaluation_seg(seg_result_gold, seg_result_pred, seg_symbol)


if __name__ == '__main__':
    gt_string = '就读 于 中国人民大学 电视 上 的 电影 节目 项目 的 研究 角色 本人 将 会 参与 配音'
    pred_string = '就读 于 中国 人民 大学 电视 上 的 电影 节目 项 目的 研究 角色 本人 将 会 参与 配音'
    sample_seg_evaluation(gt_string, pred_string)
