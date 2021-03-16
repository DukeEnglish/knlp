#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: inference
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-28
# Description:
# -----------------------------------------------------------------------#

from knlp.seq_sentiment.sentiment import SentimentAnalysis


def sentiment(sentence, function_name="sentiment_snownlp"):
    """
        This function could call different function to do sentiment analysis

    Args:
        sentence: string
        function_name: string

    Returns: float

    """
    sentiment_analyst = getattr(SentimentAnalysis, function_name, None)
    if not sentiment_analyst:
        # TODO raise an exception
        return None
    return sentiment_analyst(sentence)
