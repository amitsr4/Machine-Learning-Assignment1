import numpy as np
from scipy.spatial import distance
from nearest_neighbour import gensmallm, predictknn, learnknn
import matplotlib.pyplot as plt
import random as rnd

data = np.load('mnist_all.npz')

train1 = data['train1']
train3 = data['train3']
train4 = data['train4']
train6 = data['train6']

test1 = data['test1']
test3 = data['test3']
test4 = data['test4']
test6 = data['test6']

testLen = len(test1)+len(test3)+len(test4)+len(test6)

x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6], testLen)


def apply_knn_on_samples(k, m, corrupted):
    errors = []
    for i in range(10):
        x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], m)
        if corrupted:
            corrupt(y_train)
            corrupt(y_test)
        classifier = learnknn(k, x_train, y_train)
        pred = predictknn(classifier, x_test)
        pred = pred.reshape(1,(len(pred)))
        error = np.mean(pred != y_test)
        errors.append(error)
    errors = np.asarray(errors)
    return np.mean(errors), np.max(errors), np.min(errors)


def plot_errors(errors, x_axe):
    plt.figure()
    plt.plot(x_axe, errors, label='Average error per sample')
    plt.xlabel("Sample size")
    plt.ylabel("Average error")
    plt.suptitle("Graph1: average test error as a function of the training sample size")
    plt.legend()
    plt.show()


# apply knn on different training sample sizes and then plot it.
def task_a():
    min_errors = []
    max_errors = []
    average_errors = []
    sample_sizes = np.arange(10, 110, 10)

    for size in sample_sizes:
        print("iteration number", size)
        avg_err, max_err, min_err = apply_knn_on_samples(1, size, False)
        average_errors.append(avg_err)
        min_errors.append(min_err)
        max_errors.append(max_err)

    plt.plot(sample_sizes, average_errors, color="blue")
    plt.plot(sample_sizes, min_errors, color="green")
    plt.plot(sample_sizes, max_errors, color="red")
    sample_sizes = np.array(sample_sizes)
    min_errors = np.array(min_errors)
    max_errors = np.array(max_errors)
    average_errors = np.array(average_errors)
    plt.errorbar(sample_sizes,
                 average_errors,
                 [average_errors - min_errors, max_errors - average_errors],
                 fmt='ok', lw=1,
                 ecolor='tomato')

    plt.title("Graph1: average test error as a function of the training sample size")
    plt.legend(["Average test error", "Minimum Error", "Maximum Error"])
    plt.xlabel("Sample size")
    plt.ylabel("Average test error")
    plt.show()


def corrupt(y):
    size = len(y)
    for i in range(int(size * 0.15)):
        options = [1, 3, 4, 6]
        options.remove(y[i])
        y[i] = rnd.choice(options)

def plot_bar_of_min_and_max(min, max, x_axe):
    plt.figure()
    plt.bar([x-0.1 for x in x_axe], min, width=0.2, label='min error for each k')
    plt.bar([x+0.1 for x in x_axe], max, width=0.2, label='max error for each k')
    plt.xlabel('K')
    plt.ylabel('Average error')
    plt.suptitle("graph 4: max and min errors")
    plt.legend()
    plt.show()


# apply knn on different K's .
def task_e(corrupted):
    average_errors = []
    min_errors = []
    max_errors = []
    k = list(range(1, 12))
    for i in k:
        print("TASK E for k : ", i)
        avg_err, max_err, min_err = apply_knn_on_samples(i, 200, corrupted)
        average_errors.append(avg_err)
        min_errors.append(min_err)
        max_errors.append(max_err)

    plot_bar_of_min_and_max(min_errors,max_errors,k)
    return average_errors, np.arange(1, 12)


task_a()

meansErrorK, testSizeK = task_e(False)
plt.plot(testSizeK, meansErrorK, color="blue")
plt.legend(["Average test error"])
plt.title("Graph 2 - Error over K")
plt.xlabel("K")
plt.ylabel("Average test error")
plt.show()

meansErrorK, testSizeK = task_e(True)
plt.plot(testSizeK, meansErrorK, color="blue")
plt.legend(["Average test error"])
plt.title("Graph 3 - Error over *corrupted* K")
plt.xlabel("K")
plt.ylabel("Average test error")
plt.show()
