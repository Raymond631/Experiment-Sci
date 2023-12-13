import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from matplotlib import pyplot as plt

# 设置随机种子
np.random.seed(1447)


def autoencoder1():
    # 加载 MNIST 数据集
    (X_train, _), (X_test, _) = mnist.load_data()
    # 数据预处理
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # 将图像转换为向量
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    input_img = Input(shape=(784,))  # 输入图像形状
    encoded = Dense(32, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)

    # 编码器模型
    encoder = Model(input_img, encoded)
    # 解码器模型
    encoded_input = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

    # 预测
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    # 显示
    draw(X_test, decoded_imgs)


def autoencoder2():
    # 加载 MNIST 数据集
    (X_train, _), (X_test, _) = mnist.load_data()
    # 数据预处理
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # 添加通道维度 (1 表示灰度图)
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

    input_img = Input(shape=(28, 28, 1))
    # 编码器
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    # 卷积层
    x = MaxPooling2D((2, 2), padding='same')(x)  # 空域下采样
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # 解码
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)  # 上采样层
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')  # 编译
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

    # 预测
    decoded_imgs = autoencoder.predict(X_test)
    draw(X_test, decoded_imgs)


def autoencoder3():
    # 加载 MNIST 数据集
    (X_train, _), (X_test, _) = mnist.load_data()
    # 数据预处理
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # 添加通道维度 (1 表示灰度图)
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
    # 加入数据噪点
    noise_factor = 0.5  # 噪点因子
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    # 给测试集加入噪点
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    # 对测试集进行截取
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    # 输入图片形状
    input_img = Input(shape=(784,))
    # 编码器
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # 解码器
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # 定义模型
    autoencoder = Model(input_img, decoded)
    # 编译模型
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # 训练模型
    autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=128, shuffle=True, validation_data=(X_test_noisy, X_test))

    # 预测
    decoded_imgs = autoencoder.predict(X_test)
    # 显示
    draw(X_test_noisy, decoded_imgs)


def draw(X_test, decoded_imgs):
    # 打印图片的数量
    n = 10
    # 画布大小
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # 解码后的图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    print("自编码器-------------------------------->")
    # autoencoder1()
    print("卷积自编码器----------------------------->")
    autoencoder2()
    print("去噪自编码器----------------------------->")
    # autoencoder3()
