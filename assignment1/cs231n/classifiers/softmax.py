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
  scores = X.dot(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
        loss += -np.log( np.exp(scores[i,y[i]])/sum(np.exp(scores[i])))
            
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= X.shape[0]
  loss += reg*np.sum(W*W)
    
  def prob(W, x, index):
    p = np.exp((x.dot(W))[index])/sum(np.exp(x.dot(W)))
    return p

  for j in xrange(num_classes):
    for i in xrange(num_train):
        if (y[i] == j):
            dW[:, j] = dW[:, j] + X[i].T
        dW[:, j] = dW[:, j] - (np.exp((X[i].dot(W))[j])/sum(np.exp(X[i].dot(W))))*(X[i].T)
    dW[:, j] /= -num_train
    dW[:, j] += reg*W[:, j]
        
    
    
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]  
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis = 1) 
  prob = exp_scores[np.arange(len(exp_scores)), y]
  prob = prob/sum_exp_scores  
  prob = -np.log(prob)
  loss = sum(prob)/num_train
  loss += reg*np.sum(W*W)


  probs = (exp_scores.T/sum_exp_scores).T
  probs[np.arange(len(probs)), y] -= 1
  dW = X.T.dot(probs)
  dW /= num_train
  dW += reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

