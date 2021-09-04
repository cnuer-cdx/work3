from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc.                   #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        input_size = input_dim
        for i in range(len(hidden_dims)):
            output_size = hidden_dims[i]
            self.params['W' + str(i+1)] = np.random.randn(input_size,output_size) * weight_scale
            self.params['b' + str(i+1)] = np.zeros(output_size)
            if self.normalization:
                self.params['gamma' + str(i+1)] = np.ones(output_size)
                self.params['beta' + str(i+1)] = np.zeros(output_size)
            input_size = output_size # 下一层的输入
        # 输出层，没有BN操作
        self.params['W' + str(self.num_layers)] = np.random.randn(input_size,num_classes) * weight_scale
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = {} # 需要存储反向传播需要的参数
        cache_dropout = {}
        hidden = X
        for i in range(self.num_layers - 1):
            if self.normalization == 'batchnorm':
                hidden,cache[i+1] = affine_bn_relu_forward(hidden,
                                    self.params['W' + str(i+1)],
                                    self.params['b' + str(i+1)],
                                    self.params['gamma' + str(i+1)],
                                    self.params['beta' + str(i+1)],
                                    self.bn_params[i])
            elif self.normalization == 'layernorm':
                hidden, cache[i + 1] = affine_ln_relu_forward(hidden,
                                        self.params['W' + str(i + 1)],
                                        self.params['b' + str(i + 1)],
                                        self.params['gamma' + str(i + 1)],
                                        self.params['beta' + str(i + 1)],
                                        self.bn_params[i])
            else:
                hidden , cache[i+1] = affine_relu_forward(hidden,self.params['W' + str(i+1)],
                                                          self.params['b' + str(i+1)])
            if self.use_dropout:
                hidden , cache_dropout[i+1] = dropout_forward(hidden,self.dropout_param)
        # 最后一层不用激活层
        scores, cache[self.num_layers] = affine_forward(hidden , self.params['W' + str(self.num_layers)],
                                                       self.params['b' + str(self.num_layers)])

        # If test mode return early
        if mode == 'test':
            return scores

        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads = 0.0, {}
        loss, dS = softmax_loss(scores , y)
        # 最后一层没有relu激活层
        dhidden, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] \
            = affine_backward(dS,cache[self.num_layers])
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)] * self.params['W' + str(self.num_layers)])
        grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]

        for i in range(self.num_layers - 1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params["W" + str(i)] * self.params["W" + str(i)])
            # 倒着求梯度
            if self.use_dropout:
                dhidden = dropout_backward(dhidden,cache_dropout[i])
            if self.normalization == 'batchnorm':
                dhidden, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            elif self.normalization == 'layernorm':
                dhidden, dw, db, dgamma, dbeta = affine_ln_relu_backward(dhidden, cache[i])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            else:
                dhidden, dw, db = affine_relu_backward(dhidden, cache[i])
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db
        return loss, grads
