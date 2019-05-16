import sys
from enum import Enum
import random
import numpy as np
from scipy import stats


# enum for the sex
class Sex(Enum):
    INFANT = 0.6
    FEMALE = 0.4
    MALE = 0.2


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
    f.close()
    return data_sets


def get_y_hat(w, x):
    return np.argmax(np.dot(w, x))


def get_loss(w, x, y):
    y_hat = get_y_hat(w, x)
    return max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))


def calc_tau(w, x, y):
    # TODO check ord
    norm = 2 * np.power(np.linalg.norm(x, ord=2), 2)
    if norm == 0:
        norm = 1  # avoid from divide by 0
    return get_loss(w, x, y) / norm


def generic_update(w, y, y_hat, x, const):
    if y != y_hat:
        w[y] = [w[y][i] + const * x[i] for i in range(Data.FEATCHERS.value)]
        w[y_hat] = [w[y_hat][i] - const * x[i] for i in range(Data.FEATCHERS.value)]
    return w


def pa_train_update(w, y, x):
    y_hat = get_y_hat(w, x)
    tau = calc_tau(w, x, y)
    generic_update(w, y, y_hat, x, tau)
    return w


def perceptron_train_update(w, y, x, eta):
    y_hat = get_y_hat(w, x)
    w = generic_update(w, y, y_hat, x, eta)
    return w


# TODO check - id update in the else also
def svm_train_update(w, y, x, eta):
    y_hat = get_y_hat(w, x)
    lamda = 0.5
    one_minus_eta_lamda = 1 - eta * lamda
    if y != y_hat:
        w[y] = [one_minus_eta_lamda * w[y][i] + eta * x[i] for i in range(Data.FEATCHERS.value)]
        w[y_hat] = [one_minus_eta_lamda * w[y_hat][i] - eta * x[i] for i in range(Data.FEATCHERS.value)]
        # update the other line(/s)
        for i in range(Data.CLUSTERS.value):
            if i != y and i != y_hat:
                w[i] = [one_minus_eta_lamda * w[i][j] for j in range(Data.FEATCHERS.value)]
    else:
        for i in range(Data.CLUSTERS.value):
            w[i] = [one_minus_eta_lamda * w[i][j] for j in range(Data.FEATCHERS.value)]
    return w

# initialize 3 zeros matrixs
def init_w_for_algs():
    w_1 = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    w_2 = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    w_3 = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    return w_1, w_2, w_3


def train(x, y):
    w_perceptron, w_svm, w_pa = init_w_for_algs()
    eta_per = 1
    eta_svm = 1.5
    iterations = 20
    for i in range(iterations):
        change_ethas = 0
        # shuffle the data
        x, y = shuffle_data(x, y)
        #        x, y = random.shuffle(x, y)  # TODO check
        for x_i, y_i in zip(x, y):
            change_ethas = (change_ethas+1) % 1000
            if change_ethas == 0:  # change every 1000 samples
                eta_per *= 0.5
                eta_svm *= 0.5
            # in every iteration - update train for all three algorithems
            w_perceptron = perceptron_train_update(w_perceptron, y_i, x_i, eta_per)
            w_svm = svm_train_update(w_svm, y_i, x_i, eta_svm)
            w_pa = pa_train_update(w_pa, y_i, x_i)
    return w_perceptron, w_svm, w_pa


# def check_line(w, x):
#  result = []
# for i in range(Data.CLUSTERS.value):
#    result.append(sum([w[i][j] * x[j] for j in range(Data.FEATCHERS.value)]))
# return np.argmax(result)


def classify(w_per, w_svm, w_pa, x):
    x = normal_data(x)  # normalize data
    for line in x:
        # get the prediction for every alg and print
        y_hat_per = get_y_hat(w_per, line)
        y_hat_svm = get_y_hat(w_svm, line)
        y_hat_pa = get_y_hat(w_pa, line)
        print("perceptron: " + str(y_hat_per) + ", " + "svm: " + str(y_hat_svm) + ", " + "pa: " + str(y_hat_pa))


def shuffle_data(x, y):
    zip_x_y = list(zip(x, y))
    random.shuffle(zip_x_y)
    new_x, new_y = zip(*zip_x_y)
    return new_x, new_y


def normal_data(data):
    normal = np.array([np.array(i) for i in data])  # matrix  to array
    normal = normal.transpose()  # to normalized by columns and not rows
    for i, val in enumerate(normal):
        min_val = min(val)
        max_val = max(val)
        diff = max_val - min_val
        if diff != 0:
            normal[i] = np.divide(np.subtract(val, min_val), diff)
    return normal.transpose()  # back to the original


def TEMP_FUNC(train_x, train_y):
    cut = 3200
    x = train_x[: cut]
    y = train_y[: cut]
    test = train_x[cut:]
    test_y = train_y[cut:]
    return x, y, test, test_y


def error_rate(w, test_x, test_y, alg):
    bad = 0
    for i, line in enumerate(test_x):
        y_hat = get_y_hat(w, line)
        if y_hat != test_y[i]:
            bad += 1
    print("Error rate for " + alg + " is:", bad / len(test_y))


def main():
    train_x = parser(sys.argv[1])
    train_x = normal_data(train_x)  # normalize data
    # print("train x:\n", train_x)
    file_y = open(sys.argv[2], "r+")
    train_y = [int(float(x)) for x in file_y]
    file_y.close()
    train_x, train_y, test_x, test_y = TEMP_FUNC(train_x, train_y)

    #test_x = train_x

    #test_y = train_y

    # print("train y:\n", train_y)
    w_per, w_svm, w_pa = train(train_x, train_y)
    # print(w)
    classify(w_per, w_svm, w_pa, test_x)
    # test_x = sys.argv[2]
    error_rate(w_per, test_x, test_y, "perceptron")
    error_rate(w_svm, test_x, test_y, "svm")
    error_rate(w_pa, test_x, test_y, "pa")


if __name__ == "__main__":
    main()
