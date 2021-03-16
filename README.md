# knlp

这是一个工具包，主要实现对中文的NLP基础操作，在实现过程中，调研了网络上很多已经开源的工具包，对他们致以深深的感谢。

在coding过程中，参考学习了很多参考pkg中的编码方式，也有直接调用。如果作者感觉到被冒犯，请随时私信联系。

本pkg的主体架构参考了snownlp和textblob，因为个人认为这种实现方式对于调用方来说最方便。


pkg中提供了inference这个方法，主要是调用各种能力进行inference，seg这样的类是实现对应的功能。最后seq_upgrade，这样的pkg中有训练使用的代码，可以用来自己进行训练

最后，这个pkg还提供了很多现成的对各种nlp任务的评估方法以及相应的评估数据集（或者地址），可以供各位NLPer进行学习使用。

和现有的NLP工具包的不同点在于，本pkg提供深度学习相关的功能，并且面向中文开发，且功能很基础，适合于based on这个进行二次改造。

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

# 参考并致谢
- snownlp
- jieba
- textblob
- https://www.letiantian.me/2014-06-10-pagerank/

# 评估结果
离线评估

Clue榜单评估结果


# 开发方案
因为这不是一个有强时间节点的工作，所以我并不适合将具体的时间节点写在这里，但是其他的方案我还是应该简单设计一下。

首先，明确一下这个工程的目标，是一个即自私又无私的项目：
1. 这个项目要能可以集成现在已经有的一些能力，并很方便的调用他们，所以在api的设计上，工程的设计上一定要友好一些，且需要从用户使用的角度来考虑。
2. 另一方面，我可以借此机会把基础的算法数据结构自己都实现一次，这是一个挺长久的工作。

## 开发节点
### 第一阶段：完成基础的工程架构，通过开源能力将基础的能力具备，且可以pip安装使用 0.1.x

这个版本的核心工作在于工程架构和链路打通，在各个模块可以调用现有的开源能力进行相应的处理和输出，并且符合期望编程规范。

第一个阶段，就不对自己实现具体的算法做要求，主要是调用开源的能力实现，这里考究的是工程架构，以及一些基础的工作。同时我也要把各个模块的评估在这里写好，这样才能评估各个模块的具体的性能如何。

- [x] test模块
- [x] 各个能力的调用入口类knlp
- [x] 序列标注
    - [x] 分词：jieba
        - [x] NER：jieba
        - [x] 分词的评估
        - [ ] 分词评估相关的blog：
- [x] 信息提取：使用网络开源实现进行，进行统一的调用接口封装
- [x] 输入归一化：适合放在utils里面对输入做统一处理
- [x] 情感分析：snownlp
- [x] 相似度计算：snownlp
- [ ] check并小结

### 第二阶段：需要自己实现各个模块的非深度学习部分，这里考究的是自己对基础的算法与数据结构在NLP上的应用能力。（要求，理论写不清楚，就不准打勾）

- [ ] 序列标注：HMM，CRF、trie
  - [ ] HMM
    - [ ] train
    - [ ] inference
  - [ ]  CRF
    - [ ] train
    - [ ] inference
  - [ ] trie
    - [ ] train
    - [ ] inference
- [ ] 通用文本分类：
  - [ ] 短文本（10个字以内）
  - [ ] 长文本
- [ ] 情感分析
- [ ] 信息提取：text_rank
- [ ] 相似度计算：
  - [ ] 词语
  - [ ] 短文本
  - [ ] 长文本

### 第三阶段：需要自己实现各个模块的深度学习部分，之所以放在这里是因为这个的迭代是一个长期工作，会有很多新的模型出来，随便follow一下就会有一次更新，不如直接放在最后

- [ ] 文本生成：
  - [ ] 机器翻译：传统LSTM
  - [ ] attention 
- [ ] 文本分类
  - [ ] 
- [ ] 序列标注



# 开发节奏

0.1.x
这个版本的核心工作在于工程架构和链路打通，在各个模块可以调用现有的开源能力进行相应的处理和输出，并且符合期望编程规范。

checklist：



考虑了一下，还需要设计一下这个pkg的代码组织架构。



