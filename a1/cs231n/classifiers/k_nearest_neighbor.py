from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
          思路：逐项计算
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
            pass
            dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return dists 


    def compute_distances_one_loop(self, X):
        """
        计算 X 中每个测试点与每个训练点之间的距离
         在 self.X_train 中对测试数据使用单个循环。

        Input / Output: Same as compute_distances_two_loops
        思路： 逐行计算
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            pass
            dists[i] = np.sqrt(np.sum((self.X_train - X[i])**2, axis = 1)).reshape(1, num_train)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        计算 X 中每个测试点与每个训练点之间的距离
         在 self.X_train 中不使用显式循环。

        Input / Output: Same as compute_distances_two_loops
        思路：
        （X1 - X2）^2 = X1^2 + X2^2 - 2 * X1 * X2
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****"""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        pass
        dists += np.sum(self.X_train**2, axis = 1).reshape(1, num_train)
        dists += np.sum(X**2, axis = 1).reshape(num_test, 1)
        dists -= 2 * np.dot(X, self.X_train.T)
        dists = np.sqrt(dists)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # 找到最小的k个值的下标
            min_idxs = np.argsort(dists[i])[0:k]
            # 选择训练数据标签的值
            closest_y = self.y_train[min_idxs]
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # np.bincount()返回一个列表，每一位值为这个数在列表中出现的次数。
            # np.argmax()返回列表中最大值的下标
            y_pred[i] = np.argmax(np.bincount(closest_y))

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
