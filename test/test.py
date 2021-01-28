#!/usr/bin/python
#-*- coding:UTF-8 -*-
#-----------------------------------------------------------------------#
# File Name: test
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-27
# Description: 
#-----------------------------------------------------------------------#

from cnlp import Cnlp


def test_all():
    with open("cnlp/data/pytest_data.txt") as f:
        text = f.read()
    res = Cnlp(text)
    print(res.seg_result)


if __name__ == '__main__':
    test_all()