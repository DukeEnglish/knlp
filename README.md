# knlp

这是一个工具包，主要实现对中文的NLP基础操作，在实现过程中，调研了网络上很多已经开源的工具包，对他们致以深深的感谢。

在coding过程中，参考学习了很多参考pkg中的编码方式，也有直接调用。如果作者感觉到被冒犯，请随时私信联系。

本pkg的主体架构参考了snownlp和textblob，因为个人认为这种实现方式对于调用方来说最方便。


pkg中提供了inference这个方法，主要是调用各种能力进行inference，seg这样的类是实现对应的功能。最后seq_upgrade，这样的pkg中有训练使用的代码，可以用来自己进行训练

最后，这个pkg还提供了很多现成的对各种nlp任务的评估方法以及相应的评估数据集（或者地址），可以供各位NLPer进行学习使用。

和现有的NLP工具包的不同点在于，本pkg提供深度学习相关的功能，并且面向中文开发，且功能很基础，适合于based on这个进行二次改造。

最关键的是，这个工具包会配置有详细的文档，提供从训练到infernece以及evaluation的全套脚手架，堪称学习利器

# 安装方式
```
pip install knlp

# FROM GITHUB SOURCE CODE
pip install git+https://github.com/DukeEnglish/knlp.git
```
# 示例方法
```python
from knlp import Knlp

def test_all():
    with open("knlp/data/pytest_data.txt") as f:
        text = f.read()
    res = Knlp(text)
    print("seg_result is", res.seg_result)
    print("ner_result is", res.ner_result)
    print("sentiment score is", res.sentiment)
    print("key_words are", res.key_words)
    print("key sentences are", res.key_sentences)
    gt_string = '就读 于 中国人民大学 电视 上 的 电影 节目 项目 的 研究 角色 本人 将 会 参与 配音'
    pred_string = '就读 于 中国 人民 大学 电视 上 的 电影 节目 项 目的 研究 角色 本人 将 会 参与 配音'
    print("evaluation res are", res.evaluation_segment(gt_string, pred_string))
    abs_path_to_gold_file = ''
    abs_path_to_pred_file = ''
    gt_file_name = f'{abs_path_to_gold_file}'
    pred_file_name = f'{abs_path_to_pred_file}'
    print("evaluation file res are", res.evaluation_segment_file(gt_file_name, pred_file_name))
```
其他示例使用方法在samples中。所有的训练数据都在data中有示例数据。

# sample使用方法
1. 序列标注的训练
    1.1 首先生成训练数据，序列标注的数据处理方法在knlp/seq_labeling/data_helper.py。数据针对的是人民日报的数据。
    1.2 其次进行训练并使用samples/hmm_sample.py，进行hmm的训练
2. 信息提取（关键词、关键短语、摘要）
    2.1 samples/IE_sample.py



# 参考并致谢
- snownlp
- jieba
- textblob
- https://www.letiantian.me/2014-06-10-pagerank/

# 评估结果
离线评估

Clue榜单评估结果

