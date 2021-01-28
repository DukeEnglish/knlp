#!/usr/bin/python
#-*- coding:UTF-8 -*-
#-----------------------------------------------------------------------#
# File Name: __init__.py
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
#-----------------------------------------------------------------------#
from cnlp.seq_labeling import seg


class Cnlp(object):
    """
    This class is designed for all the basic caller


    """
    def __init__(self, data):
        self.data = data

    @property
    def seg_result(self, function_name="jieba_cut"):
        """
        This function retrun the result cut by seg defined in seq_labeling

        Args:
            function_name:

        Returns:

        """
        return seg(self.data, function_name=function_name)