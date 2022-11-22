import numpy as np
from scipy.spatial import distance
import numpy.linalg as la

class Classifier:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k


    def classify(self, x_to_classify):
        # caculates distances
        distances = []
        # for i in range(len(self.X)):
        #     distances.append([i, la.norm(self.X[i] - x_to_classify)])
        distances = np.array([(np.linalg.norm(self.X[i] - x_to_classify), self.y[i]) for i in range(len(self.X))])

        distances = distances[distances[:, 0].argsort()]  # sort by the first column
        topK = distances[:self.k]  # get only k first rows
        topK = topK[:, 1]  # get the second column
        values, counts = np.unique(topK, return_counts=True)
        ind = np.argmax(counts)  # together they bring the index of the most frequent value
        return values[ind]



def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you
"""

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
"""


# Calculate Euclidean Distance
#     def classifier(x_to_classify):
# distances = []
# for i in range(len(x_train)):
#     distances.append([i, distance.euclidean(x_train[i], x_to_classify)])
# distances.sort(key=lambda l: l[1])
# # Find the k closest examples
# distances = np.asarray(distances)
# indexes_k_closest = distances[0:k][:, 0]
# labels_of_k_closest = y_train[[int(x) for x in indexes_k_closest]]
# labels_of_k_closest = [int(x) for x in labels_of_k_closest]
#
# return np.argmax(np.bincount(labels_of_k_closest))

# return classifier
def learnknn(k: int, x_train: np.array, y_train: np.array):


    return Classifier(x_train, y_train, k)


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    # return np.reshape(np.asarray([int(classifier(x)) for x in x_test]), (len(x_test), 1))
    yPrediction = np.array([classifier.classify(x) for x in x_test])
    return yPrediction.reshape((len(yPrediction), 1))

def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
