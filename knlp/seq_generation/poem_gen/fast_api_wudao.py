# !/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: fast_api_wudao
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-09-04
# Description:
# -----------------------------------------------------------------------#
import json

import requests


def wudao_fast_poem_gen(api_key, api_secret, topic, author, speed="fast", key="queue1"):
    """
    悟道快速写诗caller function

    Args:
        api_key:
        api_secret:
        key:
        topic:
        author:
        speed:

    Returns:

    """
    request_url = "https://pretrain.aminer.cn/api/v1/"
    api = 'poem'

    # 指定请求参数格式为json
    headers = {'Content-Type': 'application/json'}
    request_url = request_url + api
    data = {
        "key": key,
        "topic": topic,
        "author": author,
        "speed": speed,
        "apikey": api_key,
        "apisecret": api_secret
    }
    response = requests.post(request_url, headers=headers, data=json.dumps(data))
    if response:
        print(response.json())
    # 返回的response里有task_id,用task_id去请求"https://gpt.aminer.cn/v1/status"接口
    '''
    请求status接口返回api调用结果
    '''
    task_id = response.json()["result"]["task_id"]  # 从之前请求api的结果中获取

    request_url = 'https://pretrain.aminer.cn/api/v1/status?task_id=' + task_id

    response = requests.get(request_url)
    if response:
        print(response.json())


if __name__ == '__main__':
    API_KEY = "ngOUvkkZQcctNYibrJT2HC1TDVeNiwE7/JO8Et7csXD8qGBthzzDNg=="
    API_SECRET = "xEWiImZgKuGqzPj/pwwdtXRMGsMjMOTZPIA1eXSoonhB1pfmzYB/Sw=="
    TOPIC = "打工人"
    AUTHOR = "李白"
    wudao_fast_poem_gen(api_key=API_KEY, api_secret=API_SECRET, topic=TOPIC, author=AUTHOR)
