import csv

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


def train(X_train, y_train):
    train_data_size = X_train.shape[0]
    # 计算数量和均值
    mu1 = np.zeros((106,))  # 类别1均值
    mu2 = np.zeros((106,))  # 类别2均值
    n1 = 0  # 类别1数量
    n2 = 0  # 类别2数量
    for i in range(train_data_size):
        if y_train[i] == 1:
            mu1 += X_train[i]
            n1 += 1
        else:
            mu2 += X_train[i]
            n2 += 1
    mu1 /= n1
    mu2 /= n2

    # 计算协方差
    sigma1 = np.zeros((106, 106))  # 类别1方差
    sigma2 = np.zeros((106, 106))  # 类别2方差
    for i in range(train_data_size):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])
    sigma1 /= n1
    sigma2 /= n2
    shared_sigma = (n1 / train_data_size) * sigma1 + (n2 / train_data_size) * sigma2  # 协方差计算

    return n1, n2, mu1, mu2, shared_sigma


def cal(X_test, n1, n2, mu1, mu2, shared_sigma):
    # 计算概率
    w = np.transpose(mu1 - mu2).dot(np.linalg.inv(shared_sigma))
    b = -0.5 * np.transpose(mu1).dot(np.linalg.inv(shared_sigma)).dot(mu1) + \
        0.5 * np.transpose(mu2).dot(np.linalg.inv(shared_sigma)).dot(mu2) + \
        np.log(float(n1 / n2))

    arr = np.empty([X_test.shape[0], 1], dtype=float)
    for i in range(X_test.shape[0]):
        z = X_test[i, :].dot(w) + b
        z *= -1
        arr[i][0] = 1 / (1 + np.exp(z))
    # 将概率限制在（0,1）之间，1e-8是一个极小数
    return np.clip(arr, 1e-8, 1 - 1e-8)


def predict(x):
    ans = np.zeros([x.shape[0], 1], dtype=int)
    # 根据计算的概率进行分类
    for i in range(len(ans)):
        if x[i] > 0.5:
            ans[i] = 1
        else:
            ans[i] = 0
    return ans


if __name__ == "__main__":
    trainData = pd.read_csv("../data/train.csv")
    testData = pd.read_csv("../data/test.csv")

    # 训练数据将107维降为106维，以适应测试数据
    X_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    y_train = dataProcess_Y(trainData).values

    X_test = dataProcess_X(testData).values

    # 计算概率所需的参数：类别1均值、类别2均值、协方差、类别1数量、类别2数量
    n1, n2, mu1, mu2, shared_sigma = train(X_train, y_train)
    # # 计算概率
    result = cal(X_test, n1, n2, mu1, mu2, shared_sigma)
    # # 判断类别
    answer = predict(result)

    ids = np.arange(1, len(answer) + 1).reshape(-1, 1)
    data_with_ids = np.concatenate((ids, answer), axis=1)
    with open("../data/predict.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        csv_writer.writerow(['id', 'label'])
        # 写入数据
        csv_writer.writerows(data_with_ids)
    print("预测结果已经保存到 data/predict.csv")
