import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

# 设置随机种子
np.random.seed(1447)


def draw(X_test, decoded_imgs):
    n = 10
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
    plt.savefig("../result/step2.png")
    plt.show()


def autoencoder2():
    # 卷积编码器
    input_img = Input(shape=(28, 28, 1))  # 输入图像形状
    ###编码器
    # 卷积层
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    # 最大池化
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # 定义一个卷积，卷积核大小为8，形状为(3,3)，激活函数为relu，padding为same
    # encoded是最大池化，池化形状为(2,2)，padding为same
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    ###解码器
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    # 上采样层
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)

    # 定义一个上采样层，形状为(2,2)
    # decoded是卷积层，卷积核大小为1，形状为(3,3)，激活函数为sigmoid，padding为same

    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.summary()
    # 数据处理
    path = '../data/mnist.npz'  # mnist数据集的文件路径
    f = np.load(path)
    X_train = f['x_train']
    X_test = f['x_test']
    f.close()

    # 数据归一化
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    # 数据形状转换
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
    print(X_train.shape)
    print(X_test.shape)

    # 训练模型
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=128,
                    shuffle=True, validation_data=(X_test, X_test), verbose=0)
    # 预测
    decoded_imgs = autoencoder.predict(X_test)
    # 显示
    draw(X_test, decoded_imgs)
