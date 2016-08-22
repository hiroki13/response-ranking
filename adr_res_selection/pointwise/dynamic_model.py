from nn import rnn, gru
from nn.nn_utils import sample_weights, build_shared_zeros, sigmoid, tanh
from nn.optimizers import ada_grad, ada_delta, adam, sgd

import theano
import theano.tensor as T


class Model(object):
    def __init__(self, c, r, a, y_r, n_r, y_a, n_a, n_agents, n_vocab, init_emb, dim_emb, dim_hidden,
                 L2_reg, unit, opt, lr, activation=tanh):

        self.tr_inputs = [c, r, a, y_r, n_r, y_a, n_a, n_agents]
        self.pr_inputs = [c, r, a, y_r, n_agents, y_a]

        self.c = c  # original c: 1D: batch, 2D: n_prev_sents, 3D: n_words
        self.r = r  # original r: 1D: batch, 2D: n_cands, 3D: n_words
        self.a = a  # 1D: batch, 2D: n_prev_sents, 3D: max_n_agents; elem=one hot vector
        self.y_r = y_r
        self.n_r = n_r
        self.y_a = y_a
        self.n_a = n_a
        self.n_agents = n_agents
        self.activation = activation

        batch = T.cast(c.shape[0], 'int32')
        n_prev_sents = T.cast(c.shape[1], 'int32')
        c_words = T.cast(c.shape[2], 'int32')
        n_cands = T.cast(r.shape[1], 'int32')
        r_words = T.cast(r.shape[2], 'int32')

        ##############
        # Parameters #
        ##############
        self.W_oa = theano.shared(sample_weights(dim_hidden * 2, dim_hidden))
        self.W_or = theano.shared(sample_weights(dim_hidden * 2, dim_hidden))

        self.pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
            self.params = [self.emb, self.W_oa, self.W_or]
        else:
            self.emb = theano.shared(init_emb)
            self.params = [self.W_oa, self.W_or]

        self.E = T.concatenate([self.pad, self.emb], 0)

        ##################
        # Initial values #
        ##################
        h_c0 = T.zeros(shape=(batch * n_prev_sents, dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * n_cands, dim_hidden), dtype=theano.config.floatX)
        A0 = T.zeros(shape=(batch, n_agents, dim_hidden), dtype=theano.config.floatX)

        ############
        # Encoders #
        ############
        if unit == 'rnn':
            c_encoder = rnn.Layer(n_i=dim_emb, n_h=dim_hidden)
            a_encoder = rnn.Transition(n_i=dim_hidden, n_h=dim_hidden)
        else:
            c_encoder = gru.Layer(n_i=dim_emb, n_h=dim_hidden)
            a_encoder = gru.Transition(n_i=dim_hidden, n_h=dim_hidden)
        self.params += c_encoder.params
        self.params += a_encoder.params

        #################
        # Lookup Tables #
        #################
        x_c = self.E[c]
        x_r = self.E[r]

        x_c = x_c.reshape((batch * n_prev_sents, c_words, dim_emb)).dimshuffle(1, 0, 2)
        x_r = x_r.reshape((batch * n_cands, r_words, dim_emb)).dimshuffle(1, 0, 2)

        ##########
        # Encode #
        ##########
        # Intra-sentential layer
        # 1D: n_words, 2D: batch * n_prev_sents, 3D: n_hidden
        H_c, _ = theano.scan(fn=c_encoder.recurrence, sequences=x_c, outputs_info=h_c0)
        h_c = H_c[-1].reshape((batch, n_prev_sents, dim_hidden)).dimshuffle(1, 0, 2)
        # 1D: n_words, 2D: batch * n_cands, 3D: n_hidden
        H_r, _ = theano.scan(fn=c_encoder.recurrence, sequences=x_r, outputs_info=h_r0)
        h_r = H_r[-1].reshape((batch, n_cands, dim_hidden))

        # Inter-sentential layer
        a = a[:, :, :n_agents].dimshuffle(1, 0, 2)
        # 1D: n_prev_sents, 2D: batch, 3D: n_agents, 4D: dim_agent
        A, _ = theano.scan(fn=a_encoder.recurrence_inter, sequences=[h_c, a], outputs_info=A0)

        ###########################
        # Extract summary vectors #
        ###########################
        # Responding Agents: 1D: Batch, 2D: dim_agent
        a_res = A[-1][:, 0]
        # Addressee Agents: 1D: batch, 2D: n_agents - 1, 3D: dim_agent
        A_adr = A[-1][:, 1:]
        # 1D: batch, 2D: dim_agent
        z = T.max(A[-1], 1)

        ###############
        # Probability #
        ###############
        o = T.concatenate([a_res, z], axis=1)
        o_a = T.dot(o, self.W_oa)  # 1D: batch, 2D: 1, 3D: dim_agent
        o_r = T.dot(o, self.W_or)  # 1D: batch, 2D: 1, 3D: dim_agent
        score_a = T.batched_dot(o_a, A_adr.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_agents-1; elem=scalar
        score_r = T.batched_dot(o_r, h_r.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_cands; elem=scalar
        self.p_a = sigmoid(score_a)
        self.p_r = sigmoid(score_r)

        ########
        # Loss #
        ########
        self.p_y_a = self.p_a[T.arange(batch), y_a]
        self.p_n_a = self.p_a[T.arange(batch), n_a]
        self.p_y_r = self.p_r[T.arange(batch), y_r]
        self.p_n_r = self.p_r[T.arange(batch), n_r]

        ######################
        # Objective Function #
        ######################
        self.nll_a = loss_function(self.p_y_a, self.p_n_a) * T.cast(T.gt(n_agents, 2), dtype=theano.config.floatX)
        self.nll_r = loss_function(self.p_y_r, self.p_n_r)
        self.nll = self.nll_r + self.nll_a
        self.L2_sqr = regularization(self.params)
        self.cost = self.nll + L2_reg * self.L2_sqr / 2.

        ################
        # Optimization #
        ################
        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

        ###########
        # Predict #
        ###########
        self.y_a_hat = T.argmax(score_a, axis=1)  # 2D: batch
        self.y_r_hat = T.argmax(score_r, axis=1)  # 1D: batch

        #####################
        # Check performance #
        #####################
        self.correct_a = T.eq(self.y_a_hat, y_a)
        self.correct_r = T.eq(self.y_r_hat, y_r)


def loss_function(y_p, n_p):
    return - (T.sum(T.log(y_p)) + T.sum(T.log(1. - n_p)))


def max_margin(xp):
    # 1D: n_queries, 2D: n_d
    query_vecs = xp[:, 0, :]  # 3D -> 2D

    # 1D: n_queries, 2D: n_cands
    scores = T.sum(query_vecs.dimshuffle((0, 'x', 1)) * xp[:, 1:, :], axis=2)
    pos_scores = scores[:, 0]
    neg_scores = scores[:, 1:]
    neg_scores = T.max(neg_scores, axis=1)

    diff = neg_scores - pos_scores + 1.0
    return T.mean((diff > 0) * diff)


def regularization(params):
    return reduce(lambda a, b: a + T.sum(b ** 2), params, 0.)

