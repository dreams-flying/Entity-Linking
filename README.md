# Entity-Linking
&emsp;&emsp;实体链接主要指的是在已有一个知识库的情况下，预测输入文本的某个实体对应知识库id。知识库里边记录了很多实体，对于同一个名字的实体可能会有多个解释，每个解释用一个唯一id编号，我们要做的就是预测文本中的实体究竟对应哪一个解释（id）。这是基于知识图谱的问答系统的必要步骤。</br>
&emsp;&emsp;在实际应用中，往往需要同时做实体识别和实体链接。实体识别的技术相对成熟了，可以参考笔者[BERT-NER](https://github.com/dreams-flying/BERT-NER)。</br>
&emsp;&emsp;本项目中笔者进行了对比实验，已知要链接的实体，只需进行实体链接即可，参见el_change.py；另外一种是先进行实体识别然后再实体链接，参见el.py。
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.9</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── data    存放数据</br>
├── chinese_embedding    存放词向量模型</br>
├── data    存放数据</br>
├── save    存放训练好的模型</br>
├── bert_embed.py bert  转化成静态词向量</br>
├── el.py    训练代码(未知实体)</br>
├── el_change.py    训练代码(已知实体)</br>
# 数据集
[百度的实体链接比赛数据](https://ai.baidu.com/broad/download?dataset=)
# 结果
|  | f1 |
| :------:| :------: |
| el_evaluate | 72.58 |
| el_change_evaluate | 87.96 |

已知实体的链接f1值高于未知实体的链接，在实际应用中，实体识别可以采用基于规则或基于深度学习的方法，那么我们只需关注链接的效果就可以了。
# 参考文献
[1]https://kexue.fm/archives/6919</br>
[2]https://github.com/bojone/el-2019
