import numpy.linalg as la
import numpy as np


class KnnClassifier():

    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def classify(self,x_to_classify):
        # caculates distances
        distances = []
        for i in range(len(self.X)):
            distances.append([i, la.norm(self.X[i] - x_to_classify)])
        distances.sort(key=lambda l: l[1])

        # build histogram of k closest x labels
        distances = np.asarray(distances)
        indexes_of_k_closest = distances[np.arange(self.k), :][:, 0]
        labels_of_k_closest = self.y[[int(x) for x in indexes_of_k_closest]]
        labels_of_k_closest = [int(x) for x in labels_of_k_closest]

        return np.argmax(np.bincount(labels_of_k_closest))


