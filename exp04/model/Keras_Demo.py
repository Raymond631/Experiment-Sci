import numpy as np
from keras import layers
from keras import models
from keras.utils import np_utils


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
    # build model
    model = models.Sequential()
    model.add(layers.Dense(10, input_shape=(28 * 28,), activation='relu'))
    model.add(layers.Activation('softmax'))
    # train model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1, validation_split=0.2)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print("test_loss:", test_loss, "\ntest_acc:", test_acc)
    return test_loss, test_acc, model


if __name__ == '__main__':
    mnist()
