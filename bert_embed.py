#! -*- coding: utf-8 -*-
#用Bert生成中文的字(词)向量，输出的字向量（词向量）是静态的

import jieba
import gensim
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from bert4keras.snippets import to_array
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model

#以词为基本单位的中文BERT（Word-based BERT） https://github.com/ZhuiyiTechnology/WoBERT
config_path = './chinese_wobert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_wobert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_wobert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# Bert字(词)向量生成
def get_data(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        char_list = f.readlines()
        char_list = char_list[5:]
        for char in char_list:
            words.append(char.replace("\n", "").strip())
    return words

def get_bert_embed(path):
    count = 0

    words = get_data(path)

    file_char_raw = open("chinese_embedding/word_embed_raw.txt", "w", encoding="utf-8")

    for word in words:
        token_ids, segment_ids = tokenizer.encode(word)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        output = model.predict([token_ids, segment_ids])
        try:
            assert len(output[0]) == 3  #jieba分词有时会将word分割，舍去该word，保留完整word和向量
        except:
            count += 1
            continue
        out_str = " ".join("%s" % embed for embed in output[0][1])
        embed_out = word + " " + out_str + "\n"
        file_char_raw.write(embed_out)
    file_char_raw.close()

    file_char = open("chinese_embedding/word_embed.txt", "w", encoding="utf-8")
    fr = open("chinese_embedding/word_embed_raw.txt", encoding="utf-8").readlines()

    file_char.write(str(len(fr)) + " " + "768" + "\n")
    for i, line in enumerate(fr):
        file_char.write(line)

    file_char.close()


    print(count)


def txt_to_bin():
    #txt文件转成bin文件
    # 预训练的词向量文件路径
    vec_path = "chinese_embedding/word_embed.txt"
    # 加载词向量文件
    wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(vec_path.replace(".txt", ".bin"))
    embed_path = "chinese_embedding/word_embed.bin"
    wv_from_text = gensim.models.KeyedVectors.load(embed_path, mmap='r')

    # 获取所有词
    vocab = wv_from_text.vocab
    # 获取所有向量
    word_embedding = wv_from_text.vectors

    # 将向量和词保存下来
    word_embed_save_path = "chinese_embedding/emebed.ckpt"
    word_save_path = "chinese_embedding/word.ckpt"
    np.save(word_embed_save_path, word_embedding)
    pd.to_pickle(vocab, word_save_path)

    # 保存word2idx和idx2word
    vocab = pd.read_pickle(word_save_path)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    pd.to_pickle(word2idx, "chinese_embedding/word2idx.ckpt")
    pd.to_pickle(idx2word, "chinese_embedding/idx2word.ckpt")


if __name__ == '__main__':

    get_bert_embed(dict_path)

    txt_to_bin()