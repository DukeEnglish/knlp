# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: wudao
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-04
# Description:
# -----------------------------------------------------------------------#
from knlp.seq_generation.poem_gen.fast_api_wudao import wudao_fast_poem_gen


def wudao_api(api_key, api_secret, topic, author, function_id=1):
    """
    This function call all the wudao api here according to its function id

    Args:
        api_key:
        api_secret:
        topic:
        author:

    Returns:

    """
    if function_id == 1:
        return wudao_fast_poem_gen(api_key=api_key, api_secret=api_secret, topic=topic, author=author)
    else:
        return None
