from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
      score = X[i].dot(W)
      #提高数值稳定性
      score -= np.max(score)
      #取出正确分类的评分
      correct_score = score[y[i]]
      exp_sum = np.sum(np.exp(score))
      loss += np.log(exp_sum) - correct_score
      for j in range(num_class):
        if j == y[i]:
          dW[:, j] += np.exp(score[j]) / exp_sum * X[i] - X[i]
        else:
          dW[:, j] += np.exp(score[j]) / exp_sum * X[i]
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    score = X.dot(W)
    #数值稳定
    score -= np.max(score, axis = 1)[:, np.newaxis]
    correct_score = score[range(num_train), y]
    exp_score = np.exp(score)
    sum_exp_score = np.sum(exp_score, axis = 1)
    loss = np.sum(np.log(sum_exp_score) - correct_score)
    loss /= num_train
    loss += reg * np.sum(W * W)
    margin = np.exp(score) / sum_exp_score.reshape(num_train, 1)
    margin[np.arange(num_train), y] += -1
    dW = X.T.dot(margin)
    dW /= num_train
    dW += 2 * reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
