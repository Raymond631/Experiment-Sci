import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# 设置随机种子
np.random.seed(1447)


def draw(X_test, decoded_imgs):
    # 打印图片的数量
    n = 10
    # 画布大小
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # 显示重构之后的图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # 保存图片文件
    plt.savefig("../result/step1.png")
    plt.show()


def autoencoder1():
    # 编码器维度
    encoding_dim = 32
    input_img = Input(shape=(784,))

    encoded = Dense(encoding_dim, activation='relu')(input_img)

    decoded = Dense(784, activation='sigmoid')(encoded)

    # 自编码模型
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # 加载数据
    path = '../data/mnist.npz'  # mnist数据集的文件路径
    f = np.load(path)
    X_train = f['x_train']
    X_test = f['x_test']
    f.close()

    # 数据归一化
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    # 形状转换
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    # print(X_train.shape)
    # print(X_test.shape)

    # 训练模型参数，补充下面代码
    # 训练5轮，batch_size设置为5，shuffle为True。
    # validation_data放入测试集数据，verbose设置为2则不输出进度条。
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=0)

    # 获取编码器和解码器
    # "encoded" 是把输入编码表示，补充下面代码
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim,))

    # "decoded" 是输入的有损重构
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

    # 预测
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    # 显示
    draw(X_test, decoded_imgs)
