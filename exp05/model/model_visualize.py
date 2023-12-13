import math

import numpy as np
from keras import backend
from keras.models import load_model
from matplotlib import pyplot as plt

from exp05.model.keras_cnn import load_data, data_preprocess, img_shape


# 权重和输出的可视化
def plot_conv_weights(weights, input_channel=0):
    # 获取权重的最高值和最低值
    w_min = np.min(weights)
    w_max = np.max(weights)
    # 卷积层中的卷积核数量
    num_filters = weights.shape[3]
    # 要绘制的网络数
    num_grids = math.ceil(math.sqrt(num_filters))
    # 创建子图
    fig, axes = plt.subplots(num_grids, num_grids)
    # 画出所有卷积核输出图像
    for (i, ax) in enumerate(axes.flat):
        # 仅画出有效卷积核图像
        if i < num_filters:
            # 获取输入通道第i个卷积核的权重
            img = weights[:, :, input_channel, i]
            # 画图
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        # 去除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# 绘制出卷积层输出的帮助函数
def plot_conv_output(values):
    #  卷积层中卷积核数量
    num_filters = values.shape[3]
    # 要绘制的网格数
    num_grids = math.ceil(math.sqrt(num_filters))
    # 创建子图
    fig, axes = plt.subplots(num_grids, num_grids)
    # 画出所有卷积核输出图像
    for (i, ax) in enumerate(axes.flat):
        # 仅画出有效卷积核图像
        if i < num_filters:
            # 获取输入通道第i个卷积核的权重
            img = values[0, :, :, i]
            # 画图
            ax.imshow(img, interpolation='nearest', cmap='binary')
        # 去除刻度线
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def model_visual(model, X_test):
    # 得到层
    model.summary()
    layer_input = model.layers[0]
    layer_conv1 = model.layers[2]
    layer_conv2 = model.layers[4]
    # 卷积权重
    weights_conv1 = layer_conv1.get_weights()[0]
    plot_conv_weights(weights_conv1, input_channel=0)
    weights_conv2 = layer_conv2.get_weights()[0]
    plot_conv_weights(weights_conv2, input_channel=0)
    # 输入图像
    image = X_test[0]
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()
    # 卷积层输出一
    output_conv1 = backend.function(inputs=[layer_input.input], outputs=[layer_conv1.output])
    layer_output1 = output_conv1([np.array([image])])[0]
    print(layer_output1.shape)
    plot_conv_output(layer_output1)
    # 卷积层输出二
    output_conv2 = backend.function(inputs=[layer_input.input], outputs=[layer_conv2.output])
    layer_output2 = output_conv2([np.array([image])])[0]
    print(layer_output2.shape)
    plot_conv_output(layer_output2)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = data_preprocess(X_train, X_test, y_train, y_test)
    # 加载模型
    model = load_model("./model.h5")
    model_visual(model, X_test)
