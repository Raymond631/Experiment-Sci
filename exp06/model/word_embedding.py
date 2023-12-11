from keras.datasets import imdb
from keras.layers import Embedding
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras_preprocessing import sequence

if __name__ == '__main__':
    max_features = 10000
    maxlen = 20
    # 加载数据
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="/home/raymond/project/Experiment-Sci/exp06/data/imdb.npz", num_words=max_features)
    # 重塑数据形状为(samples, maxlen)的二维整数张量
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    # 定义一个序列模型
    model = Sequential()
    # 添加一个Embedding层，标记个数 10000，维度 8，输入长度是maxlen
    model.add(Embedding(10000, 8, input_length=maxlen))
    # 添加一个Flatten层
    model.add(Flatten())
    # 添加一个全连接层，输出维度是1，激活函数‘sigmoid’, 作为分类器
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型，优化器选取‘rmsprop’，损失函数选取‘binary_crossentropy’,评估方式是‘acc’
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    # 拟合模型，epoch选取 10，batch_size选取 32，validation_split为 0.2
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    # 打印模型结构
    model.summary()
