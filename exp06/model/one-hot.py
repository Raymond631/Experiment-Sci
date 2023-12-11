import string

import numpy as np
from keras.preprocessing.text import Tokenizer


# 单词级的 one-hot 编码
def word_one_hot(samples):
    # x,y表示显示的第x个元素的第y个单词
    # 构建数据中所有标记的索引，用一个字典来存储
    token_index = {}
    for sample in samples:
        # 利用split方法对样本进行分词.
        for word in sample.split():
            if word not in token_index:
                # 每一个单词指定唯一索引
                token_index[word] = len(token_index)
    # 只考虑样本前max_length个单词
    max_length = 10
    # 结果返回给results:
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            # 唯一的元素为1
            results[i, j, index] = 1
    # 查看索引字典和样本列表的第二个元素的第二个单词的编码情况
    print(results[1, 1])


# 字符级的one-hot编码
def char_one_hot(samples):
    # x,y表示显示的第x个元素的第y个字符
    # 可以打印的ASCII字符
    characters = string.printable

    # 创建索引字典
    token_index = dict(zip(characters, range(len(characters))))
    # 只考虑样本前max_length个单词
    max_length = 50
    # 构建合适的 results 数组，将结果返回给results:
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, character in list(enumerate(sample[:max_length])):
            index = token_index.get(character)
            results[i, j, index] = 1

    # 查看索引字典和样本列表的第三个元素的第三个单词的编码情况
    print(results[2, 2])


if __name__ == '__main__':
    # 给 samples 列表新增一个元素‘a panda is sleeping’
    samples = ['The cat sat on the mat.',
               'The dog ate my homework.',
               'a panda is sleeping']
    word_one_hot(samples)
    char_one_hot(samples)

    # 用keras实现单词级的one-hot编码
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)
    word_index = tokenizer.word_index
    print(word_index)
