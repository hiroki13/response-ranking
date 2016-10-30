import numpy as np
import theano
import theano.tensor as T


def relu(x):
    return T.nnet.relu(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def normalize_3d(x, eps=1e-8):
    # x is len*batch*d
    # l2 is len*batch*1
    l2 = x.norm(2, axis=2).dimshuffle((0, 1, 'x'))
    return x / (l2 + eps)


def binary_cross_entropy(y_p, n_p):
    return - (T.sum(T.log(y_p)) + T.sum(T.log(1. - n_p)))


def cross_entropy_loss(scores):
    """
    :param scores: 1D: n_queries, 2D: n_cands; column0=pos, column1-=neg
    :return: avg hinge_loss: float
    """
    probs = T.nnet.softmax(scores)
    return - T.mean(T.log(probs[:, 0]))


def hinge_loss(y_p, n_p):
    """
    :param y_p: 1D: batch, 2D: dim_h
    :param n_p: 1D: batch, 2D: dim_h
    :return: float32
    """
    diff = 0.1 - y_p + n_p
    return T.mean((diff > 0) * diff)


def regularization(params):
    return reduce(lambda a, b: a + T.sum(b ** 2), params, 0.)


def get_grad_norm(grads):
    g_norm = 0.
    for g in grads:
        g_norm += g.norm(2)**2
    return T.sqrt(g_norm)


def initialize_weights(size_x, size_y=0):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-0.1,
                                         high=0.1,
                                         size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-0.1,
                                         high=0.1,
                                         size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    return W


def initialize_weights_xavier(size_x, size_y=0):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    return W


def build_shared_zeros(shape):
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        borrow=True
    )
