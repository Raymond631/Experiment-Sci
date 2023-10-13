import csv

import numpy as np


def read_train_csv():
    data = []
    for i in range(18):  # 每一行一种污染物
        data.append([])
    with open('../data/train.csv', 'r', encoding='gbk') as file:
        csv_reader = csv.reader(file)
        # 遍历每一行
        for row, row_data in enumerate(csv_reader):
            if row > 0:  # 跳过表头
                # 遍历每一列
                for col in range(3, 27):  # 不包括27
                    if row_data[col] == "NR":
                        data[(row - 1) % 18].append(float(0))
                    else:
                        data[(row - 1) % 18].append(float(row_data[col]))
    # 构建训练集
    x_train = []
    y_train = []
    for i in range(12):
        # 每个月20*24-9=471组数据
        for j in range(471):
            x_train.append([])
            # 将18行x9列的数据作为一个x
            for t in range(18):
                for s in range(9):
                    x_train[i * 471 + j].append(data[t][i * 480 + j + s])
            y_train.append(data[9][i * 20 * 24 + j + 9])  # data的第9列为PM2.5（从第0列开始算）
    # 加入偏置，在第一列添加一列1
    x_train = [[1] + row for row in x_train]
    # 将list转为numpy
    return np.array(x_train), np.array(y_train)


def train_lr(x_train, y_train):
    # 初始化参数
    weight = np.random.rand(x_train.shape[1])  # 随机参数：np.random.rand(x.shape[1])
    learning_rate = 10  # 学习率
    num_iterations = 10000  # 迭代次数
    grad_squared_sum = np.zeros(x_train.shape[1])  # Adagrad的初始梯度平方和

    # 梯度下降
    for i in range(num_iterations):
        # 计算预测值
        y_pred = np.dot(x_train, weight)

        # 计算损失
        loss = np.mean((y_pred - y_train) ** 2)  # mean()为求均值，**为乘方
        # 打印每100次迭代的损失
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

        # 计算梯度
        gradient = 2 * np.dot(x_train.T, (y_pred - y_train)) / len(y_train)
        # 更新权重
        grad_squared_sum += gradient ** 2
        weight = weight - (learning_rate / np.sqrt(grad_squared_sum + 1e-8)) * gradient

    # 保存模型参数
    np.save('model.npy', weight)


def read_test_csv():
    x_test = []
    for i in range(240):  # 240组测试数据，每组数据18*9个值
        x_test.append([])
    with open('../data/test.csv', 'r', encoding='gbk') as file:
        csv_reader = csv.reader(file)
        # 遍历每一行
        for row, row_data in enumerate(csv_reader):
            for col in range(2, 11):
                if row_data[col] == "NR":
                    x_test[int(row / 18)].append(float(0))
                else:
                    x_test[int(row / 18)].append(float(row_data[col]))
    # 加入偏置，在第一列添加一列1
    x_test = [[1] + row for row in x_test]
    # 将list转为numpy
    return np.array(x_test)


def test_lr(x_test):
    weight = np.load('model.npy')
    y_pred = np.dot(x_test, weight)
    for index, data in enumerate(y_pred):
        print(f"id_{index}的预测结果为：{data}")


def lr():
    # 训练模型
    x_train, y_train = read_train_csv()
    train_lr(x_train, y_train)
    # 测试模型
    x_test = read_test_csv()
    test_lr(x_test)


if __name__ == '__main__':
    lr()
