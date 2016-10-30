import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, initialize_weights


class Layer(object):

    def __init__(self, n_i, n_h):
        self.W = theano.shared(initialize_weights(n_i, n_h))
        self.params = [self.W]

    def forward(self, h):
        return T.dot(h, self.W)


class RNN(object):

    def __init__(self, n_i, n_h, activation=tanh):
        self.W_in = theano.shared(initialize_weights(n_i, n_h))
        self.W_h = theano.shared(initialize_weights(n_h, n_h))
        self.params = [self.W_in, self.W_h]
        self.activation = activation

    def recurrence(self, x_t, h_t):
        """
        :param x_t: 1D: batch * n_prev_sents * max_n_agents, 2D: dim_emb
        :return: h_t: 1D: batch * n_prev_sents * max_n_agents, 2D: dim_hidden
        """
        return self.activation(T.dot(x_t, self.W_in) + T.dot(h_t, self.W_h))

    def recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch, 2D: dim_hidden
        :param  i_t: 1D: batch, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch, 2D: n_agents, 3D: dim_agent
        """
        h_x = T.dot(x_t, self.W_in).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')
        h_a = T.dot(A_t, self.W_h)
        return self.activation(h_x + h_a)

    def skip_recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  i_t: 1D: batch_size, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        h_x = T.dot(x_t, self.W_in)
        A_sub = A_t[T.arange(A_t.shape[0]), i_t]
        h_a = T.dot(A_sub, self.W_h)
        return self.activation(T.set_subtensor(A_sub, h_x + h_a))


class GRU(object):

    def __init__(self, n_i, n_h, activation=tanh):
        self.W_xr = theano.shared(initialize_weights(n_i, n_h))
        self.W_hr = theano.shared(initialize_weights(n_h, n_h))
        self.W_xz = theano.shared(initialize_weights(n_i, n_h))
        self.W_hz = theano.shared(initialize_weights(n_h, n_h))
        self.W_xh = theano.shared(initialize_weights(n_i, n_h))
        self.W_hh = theano.shared(initialize_weights(n_h, n_h))
        self.params = [self.W_xr, self.W_hr, self.W_xz, self.W_hz, self.W_xh, self.W_hh]
        self.activation = activation

    def recurrence(self, x_t, h_tm1):
        r_t = sigmoid(T.dot(x_t, self.W_xr) + T.dot(h_tm1, self.W_hr))
        z_t = sigmoid(T.dot(x_t, self.W_xz) + T.dot(h_tm1, self.W_hz))
        h_hat_t = self.activation(T.dot(x_t, self.W_xh) + T.dot((r_t * h_tm1), self.W_hh))
        h_t = (1. - z_t) * h_tm1 + z_t * h_hat_t
        return h_t

    def recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  i_t: 1D: batch_size, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        x_r = T.dot(x_t, self.W_xr).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')
        x_z = T.dot(x_t, self.W_xz).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')
        x_h = T.dot(x_t, self.W_xh).dimshuffle(0, 'x', 1) * i_t.dimshuffle(0, 1, 'x')

        r_t = sigmoid(x_r + T.dot(A_t, self.W_hr))
        z_t = sigmoid(x_z + T.dot(A_t, self.W_hz))
        h_hat_t = self.activation(x_h + T.dot((r_t * A_t), self.W_hh))
        h_t = (1. - z_t) * A_t + z_t * h_hat_t

        return h_t

    def skip_recurrence_inter(self, x_t, i_t, A_t):
        """
        :param  x_t: 1D: batch_size, 2D: dim_hidden
        :param  i_t: 1D: batch_size, 2D: n_agents; elem=one hot vector
        :return A_t: 1D: batch_size, 2D: n_agents, 3D: dim_agent
        """
        x_r = T.dot(x_t, self.W_xr)
        x_z = T.dot(x_t, self.W_xz)
        x_h = T.dot(x_t, self.W_xh)

        A_sub = A_t[T.arange(A_t.shape[0]), i_t]
        r_t = sigmoid(x_r + T.dot(A_sub, self.W_hr))
        z_t = sigmoid(x_z + T.dot(A_sub, self.W_hz))
        h_hat_t = self.activation(x_h + T.dot((r_t * A_sub), self.W_hh))
        h_t = (1. - z_t) * A_sub + z_t * h_hat_t

        return T.set_subtensor(A_sub, h_t)
