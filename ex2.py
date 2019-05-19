# Tal levy 312497910
# Tomer Shlasky 204300602

import sys
from enum import Enum
import random
import numpy as np


# enum for the Gender
class Sex(Enum):
    INFANT = 0.6
    FEMALE = 0.4
    MALE = 0.2


# enum for the data
class Data(Enum):
    CLUSTERS = 3
    FEATCHERS = 8


"""
parser- split the test_x and the train_x by ","
replace the sex featcher with numeric values
takes the values from the file and write it to matrix
"""


def parser(file):
    data_sets = []
    f = open(file, "r+")  # open file for reading
    for line in f:
        if "M" in line:
            local_line = line.replace("M", str(Sex.MALE.value))
        elif "F" in line:
            local_line = line.replace("F", str(Sex.FEMALE.value))
        else:
            local_line = line.replace("I", str(Sex.INFANT.value))
        tokens = [float(x.strip()) for x in local_line.split(',')]
        data_sets.append(tokens)
    f.close()
    return data_sets


# calculate y hat
def get_y_hat(w, x):
    return np.argmax(np.dot(w, x))


# calculate the loss function
def get_loss(w, x, y):
    y_hat = get_y_hat(w, x)
    return max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))


# calculate tao for the pa algorithm
def calc_tau(w, x, y):
    norm = 2 * np.power(np.linalg.norm(x, ord=2), 2)
    if norm == 0:
        norm = 1  # avoid from divide by 0
    return get_loss(w, x, y) / norm


"""
generic update - for the pa and the pa.
checks if y != y_hat, if it is changes w (weights matrix)
by changing y and y hat columns
getting const - eta ot tau
"""


def generic_update(w, y, y_hat, x, const):
    if y != y_hat:
        w[y] = [w[y][i] + const * x[i] for i in range(Data.FEATCHERS.value)]
        w[y_hat] = [w[y_hat][i] - const * x[i] for i in range(Data.FEATCHERS.value)]
    return w


"""
pa train-
calc y hat and tao.
if y != y hat do the update.
"""


def pa_train_update(w, y, x):
    y_hat = get_y_hat(w, x)
    tau = calc_tau(w, x, y)
    generic_update(w, y, y_hat, x, tau)
    return w


"""
perceptron train-
calc y hat.
if y != y hat do the update.
"""


def perceptron_train_update(w, y, x, eta):
    y_hat = get_y_hat(w, x)
    w = generic_update(w, y, y_hat, x, eta)
    return w


"""
svm update
calc y hat and lambda
if y != y hat- change w by changing y and y hat columns
update all other columns
"""


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


# initialize 3 zeros matrix
def init_w_for_algs():
    w_1 = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    w_2 = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    w_3 = np.zeros([Data.CLUSTERS.value, Data.FEATCHERS.value])
    return w_1, w_2, w_3


"""
train all 3 algorithms - perceptron, svm and pa
init 3 0 matrix- one for every algorithm and pick constants- eta and number of iterations
in every iteration- shuffle the data and go over x (data) and y (labels)
every 1000 samples on the data - change the constants
train the 3 algorithms (calc y hat, check if y!=y hat and if it is change w)
returns the final three w-s.
"""


def train(x, y):
    w_perceptron, w_svm, w_pa = init_w_for_algs()
    eta_per = 1
    eta_svm = 1.5
    iterations = 20
    change_ethas = 0  # change every 1000 samples
    for i in range(iterations):
        # shuffle the data
        x, y = shuffle_data(x, y)
        for x_i, y_i in zip(x, y):
            change_ethas = (change_ethas + 1) % 1000
            if change_ethas == 0:  # change every 1000 samples
                eta_per *= 0.5
                eta_svm *= 0.5
            # in every iteration - update train for all three algorithems
            w_perceptron = perceptron_train_update(w_perceptron, y_i, x_i, eta_per)
            w_svm = svm_train_update(w_svm, y_i, x_i, eta_svm)
            w_pa = pa_train_update(w_pa, y_i, x_i)
    return w_perceptron, w_svm, w_pa


"""
classify (after training)
calc y hat for every sample- and display it gor every algorithm.
"""


def classify(w_per, w_svm, w_pa, x):
    for line in x:
        # get the prediction for every alg and print
        y_hat_per = get_y_hat(w_per, line)
        y_hat_svm = get_y_hat(w_svm, line)
        y_hat_pa = get_y_hat(w_pa, line)
        print("perceptron: " + str(y_hat_per) + ", " + "svm: " + str(y_hat_svm) + ", " + "pa: " + str(y_hat_pa))


"""
shuffle the data
zip x and y, shuffle and unzip them
"""


def shuffle_data(x, y):
    zip_x_y = list(zip(x, y))
    random.shuffle(zip_x_y)
    new_x, new_y = zip(*zip_x_y)
    return new_x, new_y


"""
normalize the data by columns (for every featcher)
do the min- max normalized algorithm.
"""


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


"""
for inner testing- cross validation
cross the data, train the first part and test the second
"""


def cross_validataion(train_x, train_y):
    cut = 3000
    x = train_x[: cut]
    y = train_y[: cut]
    test = train_x[cut:]
    test_y = train_y[cut:]
    return x, y, test, test_y


"""
for inner testing- calc the error rate
check the y and the y hat for every x and sum the number of mistakes
divide it by the number of examples
"""


def error_rate(w, test_x, test_y, alg):
    bad = 0
    for i, line in enumerate(test_x):
        y_hat = get_y_hat(w, line)
        if y_hat != test_y[i]:
            bad += 1
    print("Error rate for " + alg + " is:", bad / len(test_y))


# print the error rate for every algorithm
def print_error(w_per, w_svm, w_pa, test_x, test_y):
    error_rate(w_per, test_x, test_y, "perceptron")
    error_rate(w_svm, test_x, test_y, "svm")
    error_rate(w_pa, test_x, test_y, "pa")


"""
get the files -
open the 3 files, pars them into 3 matrix, 
normalized thee *_x files and close the files.
"""


def get_files():
    train_x = parser(sys.argv[1])  # get train_x
    train_x = normal_data(train_x)  # normalize train_x
    file_y = open(sys.argv[2], "r+")  # get train_y
    train_y = [int(float(x)) for x in file_y]
    file_y.close()
    test_x = parser(sys.argv[3])  # get test_x
    test_x = normal_data(test_x)  # normalize test_x
    return train_x, train_y, test_x


"""
open the 3 files.
train the perceptron, svm and pa algorithms with the given data.
get the weights matrix for every algorithm
classify the test_x data with those 3 matrix
"""


def main():
    train_x, train_y, test_x = get_files()
    w_per, w_svm, w_pa = train(train_x, train_y)
    classify(w_per, w_svm, w_pa, test_x)
    # print_error(w_per, w_svm, w_pa, test_x, train_y)


# call main
if __name__ == "__main__":
    main()
