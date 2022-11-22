import numpy as np
from scipy.spatial import distance
from nearest_neighbour1 import gensmallm, predictknn, learnknn
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

full_test_sample = [*test1, *test3, *test4, *test6]
labels = [*np.repeat(1, len(test1)), *np.repeat(3, len(test3)), *np.repeat(4, len(test4)), *np.repeat(6, len(test6))]
labels = np.reshape(np.asarray(labels, int), (len(labels), 1))


def apply_knn_on_samples(k, m):
    errors = []
    for i in range(10):
        x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], m)
        classifier = learnknn(k, x_train, y_train)
        pred = predictknn(classifier, full_test_sample)
        error = np.mean(pred != labels)
        errors.append(error)
    errors = np.asarray(errors)
    return np.min(errors), np.max(errors), np.mean(errors)


def plot_errors(errors, x_axe):
    plt.figure()
    plt.plot(x_axe, errors, label='Average error per sample')
    plt.xlabel("Sample size")
    plt.ylabel("Average error")
    plt.suptitle("Graph1: average test error as a function of the training sample size")
    plt.legend()
    plt.show()


def plot_min_max(min, max, x_axe):
    plt.figure()
    plt.bar([x - 0.5 for x in x_axe], min, label='min error for each sample size')
    plt.bar([x + 0.5 for x in x_axe], max, label='max error for each sample size')
    plt.xlabel('Sample size')
    plt.ylabel('Average error')
    plt.suptitle("Graph2: max and min errors")
    plt.legend()
    plt.show()


def plot_bar_of_min_and_max(min, max, x_axe):
    plt.figure()
    plt.bar([x - 0.5 for x in x_axe], min, label='min error for each sample size')
    # plt.bar([x-0.1 for x in x_axe], min, width=0.2, label='min error for each k')
    plt.bar([x + 0.5 for x in x_axe], max, label='max error for each sample size')
    # plt.bar([x+0.1 for x in x_axe], max, width=0.2, label='max error for each k')
    plt.xlabel('k')
    plt.xlabel('sample size')
    # plt.ylabel('average error')
    # plt.suptitle("graph4: max and min errors")
    plt.suptitle("graph2: max and min errors")
    plt.legend()
    plt.show()


# apply knn on different training sample sizes.
def task_a():
    min_errors = []
    max_errors = []
    average_errors = []
    sample_sizes = np.arange(10, 110, 10)
    for i in sample_sizes:
        min_err, max_err, avg_err = apply_knn_on_samples(1, i)
        min_errors.append(min_errors)
        max_errors.append(max_errors)
        average_errors.append(avg_err)
    plot_errors(average_errors, sample_sizes)
    # plot_min_max(min_errors, max_errors, sample_sizes)
    plot_bar_of_min_and_max(min_errors, max_errors, sample_sizes)


def corrupt(y):
    size = len(y)
    for i in range(int(size * 0.15)):
        options = [1, 3, 4, 6]
        options.remove(y[i])
        y[i] = rnd.choice(options)


# apply knn on different K's .
def task_e(corrupted):
    meansError = np.array([])
    for k in range(1, 12):
        sumMeanI = 0
        for i in range(10):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], 200)
            if corrupted:
                corrupt(y_train)
                corrupt(full_test_sample)
            classifier = learnknn(k, x_train, y_train)
            preds = predictknn(classifier, full_test_sample)
            # preds = preds.reshape(testLen, )
            sumMeanI += np.mean(labels != preds)

        meansError = np.append(meansError, [sumMeanI / 10.0])

    return meansError, np.arange(1, 12)


task_a()
# task_e(False);
# task_e(True);

# meansErrorK, testSizeK = task_e(False)
# plt.plot(testSizeK, meansErrorK, color="blue")
# plt.legend(["Average test error"])
# plt.title("Error over K")
# plt.xlabel("K")
# plt.ylabel("Average test error")
# plt.show()
