#! -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json, re, jieba, gensim
from tqdm import tqdm
import numpy as np
from random import choice
from itertools import groupby
from gensim.models import Word2Vec
from nlp_zero import Trie # pip install nlp_zero
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.callbacks import Callback
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.layers import MultiHeadAttention
from keras.layers import *
from bert4keras.backend import K

mode = 0
min_count = 2
char_size = 128
num_features = 3


# 参考bert_embed.py
embed_path = "chinese_embedding/word_embed.bin"
wv_from_text = gensim.models.KeyedVectors.load(embed_path, mmap='r')
weight_numpy = np.load(file="chinese_embedding/emebed.ckpt.npy")
word2id = pd.read_pickle("chinese_embedding/word2idx.ckpt")
id2word = pd.read_pickle("chinese_embedding/idx2word.ckpt")
word_size = weight_numpy.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), weight_numpy])
print(weight_numpy.shape)


# 词向量下载 https://kexue.fm/archives/6906

# word2vec = Word2Vec.load('./word2vec_baike/word2vec_baike')
# id2word = {i + 1: j for i, j in enumerate(word2vec.wv.index2word)}
# word2id = {j: i for i, j in id2word.items()}
# word2vec = word2vec.wv.syn0
# word_size = word2vec.shape[1]
# word2vec = np.concatenate([np.zeros((1, word_size)), word2vec])
# print(word2vec.shape)

def tokenize(s):
    """jieba分词
    """
    word_list = []
    seg_list = jieba.cut(s, HMM=False)
    for word in seg_list:
        word_list.append(word)
    return word_list

def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[(-1)].append(word2id.get(w, 0))
    V = sequence_padding(V)
    V = word2vec[V]
    return V

id2kb = {}
with open('data/raw_data/kb_data', encoding="utf-8") as (f):
    for l in tqdm(f):
        _ = json.loads(l)
        subject_id = _['subject_id']
        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        subject_alias = [alias.lower() for alias in subject_alias]
        object_regex = set(
            [i['object'] for i in _['data'] if len(i['object']) <= 10]
        )
        object_regex = sorted(object_regex, key=lambda s: -len(s))
        object_regex = [re.escape(i) for i in object_regex]
        object_regex = re.compile('|'.join(object_regex)) # 预先建立正则表达式，用来识别object是否在query出现过
        _['data'].append({
            'predicate': u'名称',
            'object': u'、'.join(subject_alias)
        })
        subject_desc = '\n'.join(
            u'%s：%s' % (i['predicate'], i['object']) for i in _['data']
        )
        subject_desc = subject_desc.lower()
        id2kb[subject_id] = {
            'subject_alias': subject_alias,
            'subject_desc': subject_desc,
            'object_regex': object_regex
        }

kb2id = {}
trie = Trie() # 根据知识库所有实体来构建Trie树

for i, j in id2kb.items():
    for k in j['subject_alias']:
        if k not in kb2id:
            kb2id[k] = []
            trie[k.strip(u'《》')] = 1
        kb2id[k].append(i)


train_data = []

with open('data/raw_data/train.json', encoding="utf-8") as (f):
    for l in tqdm(f):
        _ = json.loads(l)
        train_data.append({
            'text': _['text'],
            'mention_data': [
                (x['mention'], int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })


if not os.path.exists('data/all_chars_me.json'):
    chars = {}
    for d in tqdm(iter(id2kb.values())):
        for c in d['subject_desc']:
            chars[c] = chars.get(c, 0) + 1
    for d in tqdm(iter(train_data)):
        for c in d['text'].lower():
            chars[c] = chars.get(c, 0) + 1
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([id2char, char2id], open('data/all_chars_me.json', 'w', encoding="utf-8"))
else:
    id2char, char2id = json.load(open('data/all_chars_me.json', encoding="utf-8"))



if not os.path.exists('data/random_order_train.json'):
    random_order = range(len(train_data))
    np.random.shuffle(random_order)
    json.dump(random_order, open('data/random_order_train.json', 'w', encoding="utf-8"), indent=4)
else:
    random_order = json.load(open('data/random_order_train.json', encoding="utf-8"))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


subjects = {}

for d in train_data:
    for md in d['mention_data']:
        if md[0] not in subjects:
            subjects[md[0]] = {}
        subjects[md[0]][md[2]] = subjects[md[0]].get(md[2], 0) + 1


candidate_links = {}

for k, v in subjects.items():
    for k1 in list(v.keys()):
        if v[k1] < 2:
            del v[k1]
    if v:
        _ = set(v.keys()) & set(kb2id.get(k, []))
        if _:
            candidate_links[k] = list(_)


test_data = []

with open('data/raw_data/develop.json', encoding="utf-8") as f:
    for l in tqdm(f):
        _ = json.loads(l)
        test_data.append(_)


def isin_feature(text_a, text_b):
    y = np.zeros(len(''.join(text_a)))
    text_b = set(text_b)
    i = 0
    for w in text_a:
        if w in text_b:
            for c in w:
                y[i] = 1
                i += 1
    return y


def is_match_objects(text, object_regex):
    y = np.zeros(len(text))
    for i in object_regex.finditer(text):
        y[i.start():i.end()] = 1
    return y


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, X1V, X2V, PRES1, PRES2, Y, T = (
                [], [], [], [], [], [], [], []
            )
            for i in idxs:
                d = self.data[i]
                text = d['text'].lower()
                text_words = tokenize(text)
                text = ''.join(text_words)
                x1 = [char2id.get(c, 1) for c in text]
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                mds = {}
                for md in d['mention_data']:
                    md = (md[0].lower(), md[1], md[2])
                    if md[0] in kb2id:
                        j1 = md[1]
                        j2 = j1 + len(md[0])
                        s1[j1] = 1
                        s2[j2 - 1] = 1
                        mds[(j1, j2)] = (md[0], md[2])
                if mds:
                    j1, j2 = choice(list(mds.keys()))
                    y1 = np.zeros(len(text))
                    y1[j1:j2] = 1
                    x2 = choice(kb2id[mds[(j1, j2)][0]])
                    if x2 == mds[(j1, j2)][1]:
                        t = [1]
                    else:
                        t = [0]
                    object_regex = id2kb[x2]['object_regex']
                    x2 = id2kb[x2]['subject_desc']
                    x2_words = tokenize(x2)
                    x2 = ''.join(x2_words)
                    y2 = isin_feature(text, x2)
                    y3 = isin_feature(text_words, x2_words)
                    y4 = is_match_objects(text, object_regex)
                    y = np.vstack([y1, y2, y3, y4]).T
                    x2 = [char2id.get(c, 1) for c in x2]
                    pre_subjects = {}   #预先知道所要链接的实体
                    for i in d["mention_data"]:
                        pre_subjects[(i[1], i[1] + len(i[0]))] = i[0]

                    pres1, pres2 = np.zeros(len(text)), np.zeros(len(text))
                    for j1, j2 in pre_subjects:
                        pres1[j1] = 1
                        pres2[j2 - 1] = 1
                    X1.append(x1)
                    X2.append(x2)
                    X1V.append(text_words)
                    X2V.append(x2_words)
                    PRES1.append(pres1)
                    PRES2.append(pres2)
                    Y.append(y)
                    T.append(t)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = sequence_padding(X1)
                        X2 = sequence_padding(X2)
                        X1V = sent2vec(X1V)
                        X2V = sent2vec(X2V)
                        PRES1 = sequence_padding(PRES1)
                        PRES2 = sequence_padding(PRES2)
                        Y = sequence_padding(Y)
                        T = sequence_padding(T)
                        yield [X1, X2, X1V, X2V, PRES1, PRES2, Y, T], None
                        X1, X2, X1V, X2V, PRES1, PRES2, Y, T = (
                            [], [], [], [], [], [], [], []
                        )


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class MyBidirectional:
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer):
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, inputs):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        x, mask = inputs
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)
    def __call__(self, inputs):
        x, mask = inputs
        x_forward = self.forward_layer(x)
        x_backward = Lambda(self.reverse_sequence)([x, mask])
        x_backward = self.backward_layer(x_backward)
        x_backward = Lambda(self.reverse_sequence)([x_backward, mask])
        x = Concatenate()([x_forward, x_backward])
        x = Lambda(lambda x: x[0] * x[1])([x, mask])
        return x


x1_in = Input(shape=(None, ))
x2_in = Input(shape=(None, ))
x1v_in = Input(shape=(None, word_size))
x2v_in = Input(shape=(None, word_size))
pres1_in = Input(shape=(None, ))
pres2_in = Input(shape=(None, ))
y_in = Input(shape=(None, 1 + num_features))
t_in = Input(shape=(1, ))

x1, x2, x1v, x2v, pres1, pres2, y, t = (
    x1_in, x2_in, x1v_in, x2v_in, pres1_in, pres2_in, y_in, t_in
)

x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)

embedding = Embedding(len(id2char) + 2, char_size)
dense = Dense(char_size, use_bias=False)

x1 = embedding(x1)
x1v = dense(x1v)
x1 = Add()([x1, x1v])
x1 = Dropout(0.2)(x1)

pres1 = Lambda(lambda x: K.expand_dims(x, 2))(pres1)
pres2 = Lambda(lambda x: K.expand_dims(x, 2))(pres2)
x1 = Concatenate()([x1, pres1, pres2])
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])

x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True))([x1, x1_mask])#CuDNNLSTM


x1 = Concatenate()([x1, y])
x1 = MyBidirectional(LSTM(char_size // 2, return_sequences=True))([x1, x1_mask])#CuDNNLSTM
ys = Lambda(lambda x: K.sum(x[0] * x[1][..., :1], 1) / K.sum(x[1][..., :1], 1))([x1, y])

x2 = embedding(x2)
x2v = dense(x2v)
x2 = Add()([x2, x2v])
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = MyBidirectional(LSTM(char_size // 2, return_sequences=True))([x2, x2_mask])#CuDNNLSTM
x12 = MultiHeadAttention(8, 16)([x1, x2, x2])
x12 = Lambda(seq_maxpool)([x12, x1_mask])

x21 = MultiHeadAttention(8, 16)([x2, x1, x1])
x21 = Lambda(seq_maxpool)([x21, x2_mask])
x = Concatenate()([x12, x21, ys])
x = Dropout(0.2)(x)
x = Dense(char_size, activation='relu')(x)
pt = Dense(1, activation='sigmoid')(x)

t_model = Model([x1_in, x2_in, x1v_in, x2v_in, pres1_in, pres2_in, y_in], pt)

train_model = Model(
    [x1_in, x2_in, x1v_in, x2v_in, pres1_in, pres2_in, y_in, t_in], pt
)

pt_loss = K.mean(K.binary_crossentropy(t, pt))
loss = pt_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


def extract_items(text_in, mention_data):
    text_words = tokenize(text_in)
    text_old = ''.join(text_words)
    text_in = text_old.lower()
    _x1 = [char2id.get(c, 1) for c in text_in]
    _x1 = np.array([_x1])
    _x1v = sent2vec([text_words])
    pre_subjects = {}   #已知
    _subjects = []
    for i in mention_data:
        pre_subjects[(i[1], i[1]+len(i[0]))] = i[0]
        _subjects.append((i[0], i[1], i[1]+len(i[0])))

    _pres1, _pres2 = np.zeros([1, len(text_in)]), np.zeros([1, len(text_in)])
    for j1, j2 in pre_subjects:
        _pres1[(0, j1)] = 1
        _pres2[(0, j2 - 1)] = 1
    if _subjects:
        R = []
        _X2, _X2V, _Y = [], [], []
        _S, _IDXS = [], {}
        for _s in _subjects:
            _y1 = np.zeros(len(text_in))
            _y1[_s[1]: _s[2]] = 1
            if _s[0] in candidate_links:
                _IDXS[_s] = candidate_links.get(_s[0], [])
            else:
                _IDXS[_s] = kb2id.get(_s[0], [])
            for i in _IDXS[_s]:
                object_regex = id2kb[i]['object_regex']
                _x2 = id2kb[i]['subject_desc']
                _x2_words = tokenize(_x2)
                _x2 = ''.join(_x2_words)
                _y2 = isin_feature(text_in, _x2)
                _y3 = isin_feature(text_words, _x2_words)
                _y4 = is_match_objects(text_in, object_regex)
                _y = np.vstack([_y1, _y2, _y3, _y4]).T
                _x2 = [char2id.get(c, 1) for c in _x2]
                _X2.append(_x2)
                _X2V.append(_x2_words)
                _Y.append(_y)
                _S.append(_s)
        if _X2:
            _X2 = sequence_padding(_X2)
            _X2V = sent2vec(_X2V)
            _Y = sequence_padding(_Y)
            _X1 = np.repeat(_x1, len(_X2), 0)
            _X1V = np.repeat(_x1v, len(_X2), 0)
            _PRES1 = np.repeat(_pres1, len(_X2), 0)
            _PRES2 = np.repeat(_pres2, len(_X2), 0)
            scores = t_model.predict([_X1, _X2, _X1V, _X2V, _PRES1, _PRES2, _Y])[:, 0]
            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):
                ks = _IDXS[k]
                vs = [j[1] for j in v]
                if np.max(vs) < 0.1:
                    continue
                kbid = ks[np.argmax(vs)]
                R.append((text_old[k[1]:k[2]], k[1], kbid))
        return R
    else:
        return []



class Evaluate(Callback):
    def __init__(self):
        self.best = 0.0
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('save/best_model_el_with_s.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        pbar = tqdm()
        for d in dev_data:
            R = set(extract_items(d['text'], d["mention_data"]))
            T = set(d['mention_data'])
            A += len(R & T)
            B += len(R)
            C += len(T)
            pbar.update(1)
            f1, pr, rc = 2 * A / (B + C), A / B, A / C
            pbar.set_description('< f1: %.4f, precision: %.4f, recall: %.4f >' % (f1, pr, rc))
        pbar.close()
        return (2 * A / (B + C), A / B, A / C)


evaluator = Evaluate()
train_D = data_generator(train_data)

if __name__ == '__main__':
    train_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=10,
        callbacks=[evaluator]
    )
else:
    train_model.load_weights('save/best_model_el_with_s.weights')