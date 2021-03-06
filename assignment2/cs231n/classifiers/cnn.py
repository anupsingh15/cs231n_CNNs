from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
     
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        pass
    
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, 3, filter_size, filter_size])
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale, [np.int(H/2)*np.int(H/2)*num_filters, hidden_dim])
        self.params['b2'] = np.zeros([hidden_dim])
        self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params['b3'] = np.zeros(num_classes)
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        out_conv, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
        out_relu, cache_relu = relu_forward(out_conv)

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        out_pool, cache_pool = max_pool_forward_naive(out_relu, pool_param)
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        pass
        out_aff1, aff1_cache = affine_forward(out_pool, W2, b2)
        relu_out, relu_cache = relu_forward(out_aff1)
        out_aff2, aff2_cache = affine_forward(relu_out, W3, b3)
        scores = out_aff2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        pass
        loss, dsoft = softmax_loss(scores, y)
        loss += self.reg*0.5*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        
        dx_aff2, dw_aff2, db_aff2 = affine_backward(dsoft, aff2_cache)
        dx_relu = relu_backward(dx_aff2, relu_cache)
        dx_aff1, dw_aff1, db_aff1 = affine_backward(dx_relu, aff1_cache)
        dx_pool =  max_pool_backward_naive(dx_aff1, cache_pool)
        dx_rel = relu_backward(dx_pool, cache_relu)
        dx_conv, dw_conv, db_conv = conv_backward_naive(dx_rel, cache_conv)
        
        
        
        grads['W3'], grads['b3'] = dw_aff2 + self.reg*W3, db_aff2
        grads['W2'], grads['b2'] = dw_aff1 + self.reg*W2, db_aff1
        grads['W1'], grads['b1'] = dw_conv + self.reg*W1, db_conv
        ###########################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
