import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt


# 1.4.绘制图像的辅助函数
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


# 1.5.绘制错误分类图像的辅助函数
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = X_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = y_test[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


def mnist():
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    path = '../data/mnist.npz'  # mnist数据集的文件路径
    f = np.load(path)
    X_train, y_train = f['x_train'], f['y_train']
    X_test, y_test = f['x_test'], f['y_test']
    f.close()
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
