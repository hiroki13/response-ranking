import theano
import theano.tensor as T

from nn_utils import relu, tanh, initialize_weights


class Attention(object):

    def __init__(self, n_h, attention=1):
        if attention == 1:
            self.W1_c = theano.shared(initialize_weights(n_h, n_h))
            self.W2_r = theano.shared(initialize_weights(n_h, n_h))
            self.W1_h = theano.shared(initialize_weights(n_h, n_h))
            self.w = theano.shared(initialize_weights(n_h, ))
            self.f = self.forward
            self.params = [self.W1_c, self.W1_h, self.w, self.W2_r]
        elif attention == 2:
            self.W1_c = theano.shared(initialize_weights(n_h, n_h))
            self.W2_r = theano.shared(initialize_weights(n_h, n_h))
            self.f = self.forward_bi
            self.params = [self.W1_c, self.W2_r]
        elif attention == 3:
            self.W1_c = theano.shared(initialize_weights(n_h, n_h))
            self.W2_r = theano.shared(initialize_weights(2 * n_h, n_h))
            self.W1_h = theano.shared(initialize_weights(n_h, n_h))
            self.w = theano.shared(initialize_weights(n_h, ))
            self.w_b = theano.shared(initialize_weights(n_h, ))
            self.f = self.forward_second_order
            self.params = [self.W1_c, self.W1_h, self.w, self.w_b, self.W2_r]
        else:
            self.W1_c = theano.shared(initialize_weights(n_h, n_h))
            self.W2_r = theano.shared(initialize_weights(n_h, n_h))
            self.W1_h = theano.shared(initialize_weights(n_h, n_h))
            self.W_m = theano.shared(initialize_weights(n_h, n_h))
            self.w = theano.shared(initialize_weights(n_h, ))
            self.f = self.forward_double
            self.params = [self.W1_c, self.W1_h, self.w, self.W2_r, self.W_m]

    def forward(self, A, a_res):
        """
        :param A: 1D: batch, 2D: n_agents, 3D: dim_h
        :param a_res: 1D: batch, 2D: dim_h
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        # 1D: batch, 2D: n_agents, 3D: dim_h
        M = tanh(T.dot(A, self.W1_c) + T.dot(a_res, self.W1_h).dimshuffle(0, 'x', 1))

        # 1D: batch, 2D: n_agents
        u = T.dot(M, self.w)

        # 1D: batch, 2D: n_agents, 3D: 1
        alpha = T.nnet.softmax(u)
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch, 2D: dim_h
        r = T.sum(A * alpha, axis=1)

        # 1D: batch, 2D: dim_h
        h = relu(T.dot(r, self.W2_r))
        return h

    def forward_bi(self, A, a_res):
        """
        :param A: 1D: batch, 2D: n_agents, 3D: dim_h
        :param a_res: 1D: batch, 2D: dim_h
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        # 1D: batch, 2D: n_agents
        M = T.sum(T.dot(A, self.W1_c) * a_res.dimshuffle(0, 'x', 1), axis=2)

        # 1D: batch, 2D: n_agents, 3D: 1
        alpha = T.nnet.softmax(M)
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch, 2D: dim_h
        r = T.sum(A * alpha, axis=1)

        # 1D: batch, 2D: dim_h
        h = relu(T.dot(r, self.W2_r))
        return h

    def forward_second_order(self, A, a_res):
        """
        :param A: 1D: batch, 2D: n_agents, 3D: dim_h
        :param a_res: 1D: batch, 2D: dim_h
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        # 1D: batch, 2D: n_agents, 3D: dim_h
        M = tanh(T.dot(A, self.W1_c) + T.dot(a_res, self.W1_h).dimshuffle(0, 'x', 1))

        # 1D: batch, 2D: n_agents
        M_a = T.dot(M, self.w)

        # 1D: batch, 2D: n_agents, 3D: 1
        alpha = T.nnet.softmax(M_a)
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch, 2D: dim_h
        r_a = T.sum(A * alpha, axis=1)

        # 1D: n_agents, 2D: dim_h
        w = self.w.dimshuffle(('x', 'x', 0))
        w = T.repeat(w, M_a.shape[1], axis=1)

        # 1D: batch, 2D: n_agents, 3D: dim_h
        M_a = M_a.dimshuffle((0, 1, 'x'))
        M_a = T.repeat(M_a, M.shape[2], axis=2)
        M_b = M - T.sum(M_a * w) / self.w.norm(2)

        beta = T.nnet.softmax(T.dot(M_b, self.w_b))
        beta = beta.dimshuffle((0, 1, 'x'))

        # 1D: batch, 2D: dim_h
        r_b = T.sum(A * beta, axis=1)

        # 1D: batch, 2D: dim_h
        h = relu(T.dot(T.concatenate([r_a, r_b], axis=1), self.W2_r))
        return h

    def forward_double(self, A, a_res):
        """
        :param A: 1D: batch, 2D: n_agents, 3D: dim_h
        :param a_res: 1D: batch, 2D: dim_h
        :return: h_after: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        """

        # 1D: batch, 2D: n_agents, 3D: dim_h
        M = tanh(T.dot(A, self.W1_c) + T.dot(a_res, self.W1_h).dimshuffle(0, 'x', 1))

        # 1D: batch, 2D: n_agents
        M_a = T.dot(M, self.w)

        # 1D: batch, 2D: dim_h
        M_b = T.max(M, axis=1)
        M_b = T.dot(M_b, self.W_m)

        # 1D: batch, 2D: n_agents, 3D: 1
        alpha = T.nnet.softmax(M_a)
        alpha = alpha.dimshuffle((0, 1, 'x'))

        # 1D: batch, 2D: 1, 3D: dim_h
        beta = T.nnet.softmax(M_b)
        beta = beta.dimshuffle((0, 'x', 1))

        # 1D: batch, 2D: n_agents, 3D: dim_h
#        gamma = - (T.log(alpha) + T.log(beta))
        gamma = alpha * beta

        # 1D: batch, 2D: dim_h
        r = T.sum(A * gamma, axis=1)

        # 1D: batch, 2D: dim_h
        h = relu(T.dot(r, self.W2_r))
        return h

