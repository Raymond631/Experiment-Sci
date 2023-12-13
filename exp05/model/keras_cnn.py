import numpy as np
from keras import Input
from keras import models
from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import np_utils
from matplotlib import pyplot as plt

img_size = 28  # 图像维度
img_size_flat = 28 * 28  # 将图像重塑为一维的长度
img_shape = (28, 28)  # 重塑图像的高度和宽度的元组
img_shape_full = (28, 28, 1)  # 重塑图像的高度，宽度和深度的元组
num_classes = 10  # 类别数量
num_channels = 1  # 通道数


def load_data():
    path = '../data/mnist.npz'  # mnist数据集的文件路径
    f = np.load(path)
    X_train, y_train = f['x_train'], f['y_train']
    X_test, y_test = f['x_test'], f['y_test']
    f.close()
    return X_train, X_test, y_train, y_test


def data_preprocess(X_train, X_test, y_train, y_test):
    # 重塑数据集
    X_train = X_train.reshape([60000, 784])
    X_test = X_test.reshape([10000, 784])
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    # 归一化
    X_train = X_train / 255
    X_test = X_test / 255
    # one-hot编码
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test


def model_build(X_train, X_test, y_train, y_test):
    # 创建一个输入层，类似于TensorFlow中的feed_dict
    # 输入形状input-shape 必须是包含图像大小image_size_flat的元组
    inputs = Input(shape=(img_size_flat,))
    # 用于构建神经网络的变量。
    net = inputs
    # 输入是一个包含784个元素的扁平数组
    #  但卷积层期望图像形状是（28,28,1）
    net = Reshape(img_shape_full)(net)
    #  具有ReLU激活和最大池化的第一个卷积层。
    net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)
    #  具有ReLU激活和最大池化的第二个卷积层.
    net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)
    # 将卷积层的4级输出展平为2级，可以输入到完全连接/密集层。
    net = Flatten()(net)
    # 具有ReLU激活的第一个完全连接/密集层。
    net = Dense(128, activation='relu')(net)
    #  最后一个完全连接/密集层，具有softmax激活功能，用于分类
    net = Dense(num_classes, activation='softmax')(net)
    # 神经网络输出
    outputs = net

    # 模型编译
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(X_train, y_train, epochs=1, batch_size=128, verbose=1, validation_split=1 / 12)
    # 模型评估
    result = model.evaluate(X_test, y_test, verbose=1)
    print(model.metrics_names[0], result[0])
    print(model.metrics_names[1], result[1])
    return model


def model_predict(model, X_test, y_test):
    # 预测概率（10000行，10列）
    y_pred = model.predict(X_test)
    # 预测结果（长度10000的一维数组）
    cls_pred = np.argmax(y_pred, axis=1)
    # 真实结果（长度10000的一维数组）
    cls_true = np.argmax(y_test, axis=1)

    plot_images(X_test[0:9], cls_true[0:9], cls_pred[0:9])
    # 错分类的图片
    correct = (cls_pred == cls_true)
    plot_example_errors(cls_pred, correct, X_test, cls_true)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)  # 分为3*3个子图
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 每个子图之间的间距
    for (i, ax) in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")
        if cls_pred is None:
            xlabel = "True:{0}".format(cls_true[i])
        else:
            xlabel = "True:{0}, Pred:{1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        # 去除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_example_errors(cls_pred, correct, X_test, y_test):
    incorrect = (correct == False)
    images = X_test[incorrect]
    cls_true = y_test[incorrect]
    cls_pred = cls_pred[incorrect]
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = data_preprocess(X_train, X_test, y_train, y_test)
    model = model_build(X_train, X_test, y_train, y_test)
    model_predict(model, X_test, y_test)
    model.save("./model.h5", save_format='h5')
