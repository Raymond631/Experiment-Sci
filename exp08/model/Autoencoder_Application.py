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

    # 保存图片文件
    plt.savefig("../result/step3.png")
    plt.show()


def autoencoder3():
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

    # 加入数据噪点
    noise_factor = 0.5  # 噪点因子
    X_train_noisy = X_train + noise_factor * \
                    np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

    # 给测试集加入噪点
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    # 对测试集进行截取
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    # 输入图片形状
    input_img = Input(shape=(28, 28, 1))
    ### 编码器
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    ### 解码器
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # 定义模型
    autoencoder = Model(input_img, decoded)

    # 编译模型，优化器为adadelta，损失函数为binary_crossentropy
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # autoencoder.summary()

    # 训练模型
    autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=128, shuffle=True, validation_data=(X_test_noisy, X_test))

    # 预测
    decoded_imgs = autoencoder.predict(X_test)
    # 显示
    draw(X_test_noisy, decoded_imgs)
