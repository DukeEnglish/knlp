#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: train
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description: 
# -----------------------------------------------------------------------#

from abc import ABC


class TrainSeqLabel(ABC):
    """
    This function offers the base class to train a new model doing seq_labeling


    """

    def __init__(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def test(self):
        pass

    def train_eval_test(self):
        pass

    def _save_nodel(self):
        pass

    @staticmethod
    def load_model(model_path):
        """
        This function could load model.

        Args:
            model_path:

        Returns:

        """
        pass