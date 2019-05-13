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
    return data_sets, i


def get_y_hat(w, x):
    return np.argmax(np.dot(w, x))


def perceptron_train(x, y, number_of_datasets):
    eta = 0.001
    w = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    iterations = 10
    for i in range(iterations):
        #        x, y = random.shuffle(x, y)  # TODO check
        for x_i, y_i in zip(x, y):
            y_i_hat = get_y_hat(w, x_i)
            if y_i != y_i_hat:
                # w[y_i] = [w_i + x1 for w_i, x1 in zip(w[y_i], x_i)]
                # w[y_i_hat] = [w_i - x1 for w_i, x1 in zip(w[y_i_hat], x_i)]
                w[y_i] = [w[y_i][i] + eta * x_i[i] for i in range(Data.FEATCHERS.value)]
                w[y_i_hat] = [w[y_i_hat][i] - eta * x_i[i] for i in range(Data.FEATCHERS.value)]
                # print(w)
    return w


def check_line(w, x):
    result = []
    for i in range(Data.CLUSTERS.value):
        result.append(sum([w[i][j] * x[j] for j in range(Data.FEATCHERS.value)]))
    return np.argmax(result)


def check_data(w, x, y):
    good = 0
    i = 0
    for line, y_i in zip(x, y):
        val = check_line(w, line)
        if val != y_i:
            print("Nooooooooooo!")
        else:
            good += 1
            print("Yesss")
        i += 1
    print("good rate=", good / i)


def shuffle_data(x, y):
    constract = list(zip(x, y))
    random.shuffle(constract)
    new_x, new_y = zip(*constract)
    return new_x, new_y


def main():
    train_x, number_of_datasets = parser(sys.argv[1])
    # print("train x:\n", train_x)
    train_y = [int(float(x)) for x in open(sys.argv[2], "r+")]
    # shuffle the data
    train_x, train_y = shuffle_data(train_x, train_y)
    # print("train y:\n", train_y)
    w = perceptron_train(train_x, train_y, number_of_datasets)
    print(w)
    check_data(w, train_x, train_y)
    # test_x = sys.argv[2]


if __name__ == "__main__":
    main()
