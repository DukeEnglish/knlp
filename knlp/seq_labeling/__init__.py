#!/usr/bin/python
#-*- coding:UTF-8 -*-
#-----------------------------------------------------------------------#
# File Name: __init__.py
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
#-----------------------------------------------------------------------#

from knlp.seq_labeling.inference import seg, ner
from knlp.seq_labeling.utils import evaluation_seg, evaluation_seg_files

__all__ = ["seg", "ner", "evaluation_seg", "evaluation_seg_files"]
