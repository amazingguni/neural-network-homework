import os
import csv
import pickle
import numpy as np

def load_titanic(data_path):
    titanic_path = os.path.join(data_path, 'titanic')

    x_train, y_train = convert_to_numpy(os.path.join(titanic_path, 'train.csv'))
    x_test, y_test = convert_to_numpy(os.path.join(titanic_path, 'test.csv'))

    return x_train, y_train, x_test, y_test


def convert_to_numpy(filename):
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)

        x_data = []
        y_data = []
        for line in csv_reader:
            features = line[1:]
            x = [1] + list(map(float, features))
            y = float(line[0])
            x_data.append(x)
            y_data.append(y)

        x_array = np.array(x_data)
        y_array = np.array(y_data).reshape(-1, 1)

    return x_array, y_array

def load_cifar100(data_path):
    cifar100_path = os.path.join(data_path, 'cifar100')

    f1 = os.path.join(cifar100_path, 'train')
    f2 = os.path.join(cifar100_path, 'test')

    with open(f1, 'rb') as f1:
        train_data = pickle.load(f1, encoding='latin1')
        x_train = train_data['data'] / 255.0
        y_train = np.asarray(train_data['fine_labels'])
        x_train = x_train.astype('float32')
        y_train = y_train.astype('int32')

    with open(f2, 'rb') as f2:
        test_data = pickle.load(f2, encoding='latin1')

        x_test = test_data['data'] / 255.0
        y_test = np.asarray(test_data['fine_labels'])
        x_test = x_test.astype('float32')
        y_test = y_test.astype('int32')


    # Flatten X
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    # Y as one-hot
    y_train = np.eye(100)[y_train]
    y_test = np.eye(100)[y_test]

    return x_train, y_train, x_test, y_test


def load_iris(data_path):
    iris_path = os.path.join(data_path, 'iris')

    x_train = np.load(os.path.join(iris_path, 'iris_train_x.npy'))
    y_train = np.load(os.path.join(iris_path, 'iris_train_y.npy'))
    x_test = np.load(os.path.join(iris_path, 'iris_test_x.npy'))
    y_test = np.load(os.path.join(iris_path, 'iris_test_y.npy'))

    y_train[y_train == 1] = -1
    y_test[y_test == 1] = -1
    y_train[y_train == 0] = 1
    y_test[y_test == 0] = 1

    return x_train, y_train, x_test, y_test

def load_fashion_mnist(data_path):
    cifar_path = os.path.join(data_path, 'fashion_mnist')

    x_train = np.load(os.path.join(cifar_path, 'fashion_train_x.npy'))
    y_train = np.load(os.path.join(cifar_path, 'fashion_train_y.npy'))
    x_test = np.load(os.path.join(cifar_path, 'fashion_test_x.npy'))
    y_test = np.load(os.path.join(cifar_path, 'fashion_test_y.npy'))

    # Flatten X
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    # Y as one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test



