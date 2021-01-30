# cnlp

这是一个工具包，主要实现对中文的NLP基础操作，在实现过程中，调研了网络上很多已经开源的工具包，对他们致以深深的感谢。

在coding过程中，参考学习了很多参考pkg中的编码方式，也有直接调用。如果作者感觉到被冒犯，请随时私信联系。

本pkg的主体架构参考了snownlp和textblob，因为这种实现方式对于调用方来说最方便。


pkg中提供了inference这个方法，主要是调用各种能力进行inference，seg这样的类是实现对应的功能。最后seq_upgrade，这样的pkg中有训练使用的代码，可以用来自己进行训练

最后，这个pkg还提供了很多现成的对各种nlp任务的评估方法以及相应的评估数据集（或者地址），可以供各位NLPer进行学习使用。

和现有的NLP工具包的不同点在于，本pkg提供深度学习相关的功能，并且面向中文开发，且功能很基础，适合于based on这个进行二次改造。

# 安装方式
```
pip install cnlp
```

# 参考并致谢
- snownlp
- jieba
- textblob
- https://www.letiantian.me/2014-06-10-pagerank/

# 评估结果
离线评估

c榜单评估结果