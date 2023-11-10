import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dataProcess_X(data):
    # income列是Y值，要从X中去掉; sex列的值可以直接使用1位二进制码表示，不需要进行one-hot编码
    if "income" in data.columns:
        Data = data.drop(["income", "sex"], axis=1)
    else:
        Data = data.drop(["sex"], axis=1)

    # 离散属性列
    listObjectData = [col for col in Data.columns if Data[col].dtypes == "object"]
    ObjectData = Data[listObjectData]
    ObjectData = pd.get_dummies(ObjectData)  # one-hot编码，7列变成100列
    ObjectData.insert(0, "sex", (data["sex"] == " Female").astype("int64"))  # 插入sex列，0代表male，1代表female
    # 连续属性列
    listNonObjectData = [col for col in Data.columns if col not in listObjectData]
    NonObjectData = Data[listNonObjectData]  # 6列
    # 合并离散属性和连续属性
    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data = Data.astype("int64")  # 一共110列
    # 数据标准化
    Data = (Data - Data.mean()) / Data.std()  # 0-1标准化：输入数据减去均值再除上标准差
    return Data


def dataProcess_Y(data):
    # income属性，0代表小于等于50K，1代表大于50K
    return (data["income"] == " >50K").astype("int64")


def normalize_column(X, train=True, specified_column=None, X_mean=True, X_std=True):
    # 归一化，将指定列的数据归一到0附近，且符合正态分布
    if train:
        if specified_column == None:
            # 如果没有指定列，则对全部列进行归一化处理
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column], 0), (1, length))
        X_std = np.reshape(np.std(X[:, specified_column], 0), (1, length))

        X[:, specified_column] = np.divide(
            np.subtract(X[:, specified_column], X_mean), X_std)

    return X, X_mean, X_std


def train_dev_split(X, y, dev_size=0.25):
    # 按照dev_size比例分割数据，用于交叉验证
    train_len = int(round(len(X) * (1 - dev_size)))
    return X[0:train_len], y[0:train_len], X[train_len:], y[train_len:]


def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def get_prob(X, w, b):
    return _sigmoid(np.add(np.matmul(X, w), b))


def infer(X, w, b):
    return np.round(get_prob(X, w, b))


def gradient(X, y, w, b):
    # 梯度计算
    y_pre = get_prob(X, w, b)
    pre_error = y - y_pre
    w_grad = -np.sum(np.multiply(pre_error, X.T), 1)
    b_grad = -np.sum(pre_error)
    return w_grad, b_grad


def gradient_regularization(X, y, w, b, lamda):
    # 进行正则化的梯度计算
    y_pre = get_prob(X, w, b)
    pre_error = y - y_pre
    w_grad = -np.sum(np.multiply(pre_error, X.T), 1) + lamda * w
    b_grad = -np.sum(pre_error)
    return w_grad, b_grad


def _cross_entropy(y, y_pre):
    cross_entropy = -np.dot(y, np.log(y_pre)) - \
                    np.dot((1 - y), np.log(1 - y_pre))
    return cross_entropy


def compute_loss(y, y_pre, lamda, w):
    return _cross_entropy(y, y_pre) + lamda * np.sum(np.square(w))


def _shuffle(X, y):
    # 打乱数据顺序
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], y[randomize]


def accuracy(y_pre, y):
    acc = np.sum(y_pre == y) / len(y_pre)
    return acc


if __name__ == '__main__':
    # 加载训练数据集
    train_data = pd.read_csv("../data/train.csv")
    # 训练数据将107维降为106维，以适应测试数据
    X_train = dataProcess_X(train_data).drop(['native_country_ Holand-Netherlands'], axis=1).values
    y_train = dataProcess_Y(train_data).values

    col = [0, 1, 3, 4, 5, 7, 10, 12, 25, 26, 27, 28]
    X_train, X_mean, X_std = normalize_column(X_train, specified_column=col)

    # 分割数据为训练集和验证集
    X_train, y_train, X_dev, y_dev = train_dev_split(X_train, y_train)
    num_train = len(y_train)  # 训练集大小
    num_dev = len(y_dev)  # 验证集大小

    max_iter = 40  # 最大迭代次数
    batch_size = 32  # 每一次迭代训练的数据量
    learning_rate = 0.01  # 学习率

    loss_train = []  # 训练误差
    loss_validation = []  # 验证误差
    acc_train = []  # 训练准确率
    acc_validation = []  # 验证准确率

    w = np.zeros((X_train.shape[1],))
    b = np.zeros((1,))

    # 正则化
    regularize = True
    if regularize:
        lamda = 0.01
    else:
        lamda = 0

    # 完善二分类模型
    for epoch in range(max_iter):
        X_train, y_train = _shuffle(X_train, y_train)

        step = 1  # 更新学习率
        # 逻辑回归
        for i in range(int(np.floor(len(y_train) / batch_size))):
            X = X_train[i * batch_size:(i + 1) * batch_size]
            y = y_train[i * batch_size:(i + 1) * batch_size]

            # 计算梯度
            w_grad, b_grad = gradient_regularization(X, y, w, b, lamda)

            # 更新w、b
            w = w - learning_rate / np.square(step) * w_grad
            b = b - learning_rate / np.square(step) * b_grad

            step = step + 1

        # 计算训练集的误差和准确率
        y_train_pre = get_prob(X_train, w, b)
        acc_train.append(accuracy(np.round(y_train_pre), y_train))
        loss_train.append(compute_loss(
            y_train, y_train_pre, lamda, w) / num_train)

        # 计算验证集的误差和准确率
        y_dev_pre = get_prob(X_dev, w, b)
        acc_validation.append(accuracy(np.round(y_dev_pre), y_dev))
        loss_validation.append(compute_loss(
            y_dev, y_dev_pre, lamda, w) / num_dev)

    test_data = pd.read_csv("../data/test.csv")
    X_test = dataProcess_X(test_data)
    features = X_test.columns.values
    X_test = X_test.values

    X_test, _, _ = normalize_column(
        X_test, train=False, specified_column=col, X_mean=X_mean, X_std=X_std)

    result = infer(X_test, w, b)

    # 输出贡献最大的10个特征
    ind = np.argsort(np.abs(w))[::-1]
    for i in ind[0:10]:
        print(features[i], w[i])

    with open("../result/predict.csv", "w+") as csvfile:
        csvfile.write("id,label\n")
        print("id, label")
        for i, label in enumerate(result):
            csvfile.write("%d,%d\n" % (i + 1, label))
            if i < 10:
                print(i + 1, ", ", label)

    plt.plot(loss_train)
    plt.plot(loss_validation)
    plt.legend(['train', 'validation'])
    plt.savefig('../data/test.jpg')

    plt.plot(acc_train)
    plt.plot(acc_validation)
    plt.legend(['train', 'validation'])
    plt.savefig('../data/test2.jpg')
