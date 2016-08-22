from nn import rnn, gru
from nn.nn_utils import sample_weights, build_shared_zeros, sigmoid, tanh
from nn.optimizers import ada_grad, ada_delta, adam, sgd

import theano
import theano.tensor as T


class Model(object):
    def __init__(self, c, r, a, y_r, n_r, y_a, n_a, n_agents, opt, lr, init_emb, dim_emb, dim_hidden,
                 n_vocab, max_n_agents, L2_reg, unit, activation=tanh):

        self.tr_inputs = [c, r, a, y_r, n_r, y_a, n_a, n_agents]
        self.pr_inputs = [c, r, a, y_r, n_agents, y_a]

        self.c = c  # original c: 1D: batch, 2D: n_prev_sents, 3D: n_words
        self.r = r  # original r: 1D: batch, 2D: n_cands, 3D: n_words
        self.a = a  # 1D: batch, 2D: n_prev_sents, 3D: max_n_agents; elem=1/0 (elem of one hot vector)
        self.y_r = y_r
        self.n_r = n_r
        self.y_a = y_a
        self.n_a = n_a
        self.max_n_agents = max_n_agents
        self.activation = activation

        batch = T.cast(c.shape[0], 'int32')
        n_prev_sents = T.cast(c.shape[1], 'int32')
        c_words = T.cast(c.shape[2], 'int32')
        n_cands = T.cast(r.shape[1], 'int32')
        r_words = T.cast(r.shape[2], 'int32')

        """ Parameters """
        self.pad = build_shared_zeros((1, dim_emb))
        self.A = theano.shared(sample_weights(max_n_agents, dim_emb))
        self.W_a = theano.shared(sample_weights(dim_emb + dim_hidden, dim_emb))
        self.W_r = theano.shared(sample_weights(dim_emb + dim_hidden, dim_hidden))

        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
            self.params = [self.emb, self.A, self.W_a, self.W_r]
        else:
            self.emb = theano.shared(init_emb)
            self.params = [self.A, self.W_a, self.W_r]

        self.E = T.concatenate([self.pad, self.emb], 0)

        """ Initial values """
        h_c0 = T.zeros(shape=(batch, dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * n_cands, dim_hidden), dtype=theano.config.floatX)

        """ Encoders """
        if unit == 'rnn':
            context_encoder = rnn.Layer(n_i=dim_emb, n_h=dim_hidden)
        else:
            context_encoder = gru.Layer(n_i=dim_emb, n_h=dim_hidden)
        self.params.extend(context_encoder.params)

        """ Look up table """
        x_c = self.E[c]  # 1D: batch, 2D: n_prev_sents, 3D: c_words, 4D: dim_emb
        x_r = self.E[r]  # 1D: batch, 2D: n_cands, 3D: r_words, 4D: dim_emb
        x_a = T.dot(a, self.A).reshape((batch, n_prev_sents, 1, dim_emb))

        context = T.concatenate([x_a, x_c], 2).reshape((batch,  n_prev_sents * (c_words + 1), dim_emb)).dimshuffle(1, 0, 2)
        response = x_r.reshape((batch * n_cands, r_words, dim_emb)).dimshuffle(1, 0, 2)

        """ Encode """
        # H_c: 1D: c_words, 2D: batch, 3D: dim_hidden
        # H_r: 1D: r_words, 2D: batch * n_cands, 3D: dim_hidden
        H_c, _ = theano.scan(fn=context_encoder.recurrence, sequences=context, outputs_info=h_c0)
        H_r, _ = theano.scan(fn=context_encoder.recurrence, sequences=response, outputs_info=h_r0)

        """ Extract summary vectors """
        a_res = T.repeat(self.A[0].dimshuffle('x', 0), batch, 0)
        h_c = H_c[-1]
        A_p = self.A[1:n_agents]  # 1D: batch, 2D: n_agents - 1, 3D: dim_emb
        h_q = H_r[-1].reshape((batch, n_cands, dim_hidden))

        """ Probability """
        h = T.concatenate([a_res, h_c], 1)  # 1D: batch, 2D: dim_emb + dim_hidden
        score_a = T.dot(T.dot(h, self.W_a), A_p.T)
        score_r = T.batched_dot(T.dot(h, self.W_r), h_q.dimshuffle(0, 2, 1))
        self.p_a = sigmoid(score_a)
        self.p_r = sigmoid(score_r)
        self.p_y_a = self.p_a[T.arange(batch), y_a]
        self.p_n_a = self.p_a[T.arange(batch), n_a]
        self.p_y_r = self.p_r[T.arange(batch), y_r]
        self.p_n_r = self.p_r[T.arange(batch), n_r]

        """ Objective Function """
        self.nll_a = loss_function(self.p_y_a, self.p_n_a) * T.cast(T.gt(n_agents, 2), dtype=theano.config.floatX)
        self.nll_r = loss_function(self.p_y_r, self.p_n_r)
        self.nll = self.nll_r + self.nll_a
        self.L2_sqr = regularization(self.params)
        self.cost = self.nll + L2_reg * self.L2_sqr / 2.

        """ Optimization """
        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

        """ Predicts """
        self.y_a_hat = T.argmax(score_a, axis=1)  # 2D: batch
        self.y_r_hat = T.argmax(score_r, axis=1)  # 1D: batch

        """ Check Accuracies """
        self.correct_a = T.eq(self.y_a_hat, y_a)
        self.correct_r = T.eq(self.y_r_hat, y_r)


def loss_function(y_p, n_p):
    return - (T.sum(T.log(y_p)) + T.sum(T.log(1. - n_p)))


def regularization(params):
    return reduce(lambda a, b: a + T.sum(b ** 2), params, 0.)
