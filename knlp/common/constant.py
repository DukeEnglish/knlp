# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: constant
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-03-20
# Description:
# -----------------------------------------------------------------------#
from pathlib import Path

KNLP_PATH = f"{Path.cwd()}"
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
