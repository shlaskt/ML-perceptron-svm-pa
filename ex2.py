import sys
from enum import Enum
import random
import numpy as np
from scipy import stats


# enum for the sex
class Sex(Enum):
    INFANT = 1
    FEMALE = 2
    MALE = 3


class Group(Enum):
    FIRST = 0
    SECOND = 1
    THIRD = 2


class Data(Enum):
    CLUSTERS = 3
    FEATCHERS = 8


def parser(file):
    data_sets = []
    f = open(file, "r+")
    i = 0
    for line in f:
        if "M" in line:
            local_line = line.replace("M", str(Sex.MALE.value))
        elif "F" in line:
            local_line = line.replace("F", str(Sex.MALE.value))
        else:
            local_line = line.replace("I", str(Sex.MALE.value))
        tokens = [float(x.strip()) for x in local_line.split(',')]
        data_sets.append(tokens)
        # TODO axis? ddof?
        # stats.mstats.zscore(data_sets[i])
        i += 1
    f.close()
    return data_sets, i


def get_y_hat(w, x):
    return np.argmax(np.dot(w, x))


def get_loss(w, x, y):
    y_hat = get_y_hat(w, x)
    return max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))


def calc_tau(w, x, y):
    # TODO check ord
    return get_loss(w, x, y) / (2 * np.power(np.linalg.norm(x, ord=2), 2))


def generic_update(w, y, y_hat, x, const):
    if y != y_hat:
        w[y] = [w[y][i] + const * x[i] for i in range(Data.FEATCHERS.value)]
        w[y_hat] = [w[y_hat][i] - const * x[i] for i in range(Data.FEATCHERS.value)]


def pa_train_update(w, y, y_hat, x, eta):
    tau = calc_tau(w, x, y)
    generic_update(w, y, y_hat, x, tau)


def perceptron_train_update(w, y, y_hat, x, eta):
    generic_update(w, y, y_hat, x, eta)


def svm_train_update(w, y, y_hat, x, eta):
    lamda = 0.5
    one_minus_eta_lamda = 1 - eta * lamda
    if y != y_hat:
        w[y] = [one_minus_eta_lamda * w[y][i] + eta * x[i] for i in range(Data.FEATCHERS.value)]
        w[y_hat] = [one_minus_eta_lamda * w[y_hat][i] - eta * x[i] for i in range(Data.FEATCHERS.value)]
        # update the other line(/s)
        for i in range(Data.CLUSTERS.value):
            if i != y and i != y_hat:
                w[i] = [one_minus_eta_lamda * w[i][j] for j in range(Data.FEATCHERS.value)]


def train(x, y, update):
    eta = 0.99
    w = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    iterations = 10
    for i in range(iterations):
        #        x, y = random.shuffle(x, y)  # TODO check
        for x_i, y_i in zip(x, y):
            y_i_hat = get_y_hat(w, x_i)
            update(w, y_i, y_i_hat, x_i, eta)
        eta /= 10
    return w


def check_line_perceptron(w, x):
    result = []
    for i in range(Data.CLUSTERS.value):
        result.append(sum([w[i][j] * x[j] for j in range(Data.FEATCHERS.value)]))
    return np.argmax(result)


def classify_perceptron(w, x, y):
    bad = 0
    i = 0
    for line, y_i in zip(x, y):
        val = check_line_perceptron(w, line)
        if val != y_i:
            bad += 1
        i += 1
    print("bad rate=", bad / i)


def shuffle_data(x, y):
    constract = list(zip(x, y))
    random.shuffle(constract)
    new_x, new_y = zip(*constract)
    return new_x, new_y


def main():
    train_x, number_of_datasets = parser(sys.argv[1])
    # print("train x:\n", train_x)
    file_y = open(sys.argv[2], "r+")
    train_y = [int(float(x)) for x in file_y]
    file_y.close()
    # shuffle the data
    train_x, train_y = shuffle_data(train_x, train_y)
    # print("train y:\n", train_y)
    w = train(train_x, train_y, perceptron_train_update)
    print(w)
    classify_perceptron(w, train_x, train_y)
    # test_x = sys.argv[2]


if __name__ == "__main__":
    main()
