# -*-coding:utf-8-*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import time


def get_dataset():
    """
    获取数据集
    :return:
    """
    data = load_iris()
    x, y = data['data'], data['target']
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.2, random_state=2019)  # 保证运行不变性
    return x_train_, y_train_, x_test_, y_test_


def test_accuracy():
    """
    测试准确率
    :return:
    """
    x_train, y_train, x_test, y_test = get_dataset()
    nnb = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(x_train)
    distance, index = nnb.kneighbors([x_test[0]])
    print("nearest distance", distance)
    print("nearest index", index)
    print("true label", y_test[0])
    print("nearest label", y_train[index])


def test_time():
    """
    测试邻居查找时间
    :return:
    """
    x_train, y_train, x_test, y_test = get_dataset()
    nnb = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(x_train)
    start_time = time.time()
    for sample in x_test:
        distance, index = nnb.kneighbors([sample])
    end_time = time.time()
    print("Spend:{} s".format(end_time-start_time))


if __name__ == '__main__':
    test_accuracy()
    test_time()