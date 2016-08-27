import os
import time
import math

import numpy as np
import theano
import theano.tensor as T

from ..utils import say, dump_data
from eval import Evaluator
from ..nn import rnn, attention
from ..nn import initialize_weights, build_shared_zeros, sigmoid, loss_function, regularization, get_grad_norm, optimizer_select


class Model(object):

    def __init__(self, argv, max_n_agents, n_vocab, init_emb):
        self.argv = argv

        ###################
        # Input variables #
        ###################
        self.inputs = []
        self.params = []

        ###################
        # Hyperparameters #
        ###################
        self.n_cands = argv.n_cands
        self.n_words = argv.max_n_words
        self.max_n_agents = max_n_agents
        self.n_vocab = n_vocab
        self.init_emb = init_emb
        self.opt = argv.opt
        self.lr = argv.lr
        self.dim_emb = argv.dim_emb if init_emb is None else len(init_emb[0])
        self.dim_hidden = argv.dim_hidden
        self.L2_reg = argv.reg
        self.unit = argv.unit
        self.attention = argv.attention

        ####################
        # Output variables #
        ####################
        self.a_hat = None
        self.r_hat = None
        self.nll = None
        self.cost = None
        self.g_norm = None
        self.update = None

    def predict(self, p_a, p_r):
        self.a_hat = T.argmax(p_a, axis=1)
        self.r_hat = T.argmax(p_r, axis=1)

    def objective_f(self, p_a, p_r, y_a, y_r, n_agents):
        # y_a: 1D: batch, 2D: max_n_agents; int32
        # p_a: 1D: batch, 2D: n_agents-1; elem=scalar

        # p_arranged: 1D: batch, 2D: n_agents-1; C0=True, C1-Cn=False
        mask = T.cast(T.neq(y_a, -1), theano.config.floatX)
        p_a_arranged = p_a.ravel()[y_a.ravel()]
        p_a_arranged = p_a_arranged.reshape((y_a.shape[0], y_a.shape[1]))
        p_a_arranged = p_a_arranged * mask

        p_r_arranged = p_r.ravel()[y_r.ravel()]
        p_r_arranged = p_r_arranged.reshape((y_r.shape[0], y_r.shape[1]))

        true_p_a = p_a_arranged[:, 0]
        false_p_a = T.max(p_a_arranged[:, 1:], axis=1)
        true_p_r = p_r_arranged[:, 0]
        false_p_r = T.max(p_r_arranged[:, 1:], axis=1)

        ########
        # Loss #
        ########
        nll_a = loss_function(true_p_a, false_p_a) * T.cast(T.gt(n_agents, 2), dtype=theano.config.floatX)
        nll_r = loss_function(true_p_r, false_p_r)
        nll = nll_r + nll_a

        ########
        # Cost #
        ########
        L2_sqr = regularization(self.params)
        cost = nll + self.L2_reg * L2_sqr / 2.

        return nll, cost

    def optimization(self):
        grads = T.grad(self.cost, self.params)
        self.g_norm = get_grad_norm(grads)
        self.update = optimizer_select(self.opt, self.params, grads, self.lr)

    def get_pnorm_stat(self):
        lst_norms = []
        for p in self.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def evaluate(self, pred_f, pred_r_f, c, r, a, y_r, y_a, n_agents):
        pred_a = None
        if n_agents > 1:
            pred_a, pred_r = pred_f(c, r, a, y_r, y_a, n_agents)
        else:
            pred_r = pred_r_f(c, r, a, y_r, y_a, n_agents)
        return pred_a, pred_r

    def evaluate_all(self, samples, pred_f, pred_r_f):
        evaluator = Evaluator()
        start = time.time()

        for i, sample in enumerate(samples):
            if i != 0 and i % 100 == 0:
                say("  {}/{}".format(i, len(samples)))

            x = sample[0]
            binned_n_agents = sample[1]
            labels_a = sample[2]
            labels_r = sample[3]
            pred_a, pred_r = self.evaluate(pred_f, pred_r_f, c=x[0], r=x[1], a=x[2], y_r=x[3], y_a=x[4], n_agents=x[5])

            evaluator.update(binned_n_agents, 0., 0., pred_a, pred_r, labels_a, labels_r)

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        evaluator.show_results()
        return evaluator.acc_both

    def train(self, train_samples, n_train_batches, evalset, dev_samples, test_samples):
        argv = self.argv
        n_prev_sents = argv.n_prev_sents
        max_n_words = argv.max_n_words

        #################
        # Build a model #
        #################
        say('\n\nBuilding a model...')
        index = T.iscalar('index')
        train_f = theano.function(inputs=[index],
                                  outputs=[self.nll, self.g_norm, self.a_hat, self.r_hat],
                                  updates=self.update,
                                  givens={
                                      self.inputs[0]: train_samples[0][index],
                                      self.inputs[1]: train_samples[1][index],
                                      self.inputs[2]: train_samples[2][index],
                                      self.inputs[3]: train_samples[3][index],
                                      self.inputs[4]: train_samples[4][index],
                                      self.inputs[5]: train_samples[5][index]
                                  }
                                  )

        pred_f = theano.function(inputs=self.inputs,
                                 outputs=[self.a_hat, self.r_hat],
                                 on_unused_input='ignore'
                                 )
        pred_r_f = theano.function(inputs=self.inputs,
                                   outputs=self.r_hat,
                                   on_unused_input='ignore'
                                   )

        ############
        # Training #
        ############
        say('\nTraining start\n')

        acc_history = {}
        best_dev_acc_both = 0.
        unchanged = 0

        cand_indices = range(n_train_batches)

        for epoch in xrange(argv.epoch):
            evaluator = Evaluator()

            say('\n\n')
            os.system("ps -p %d u" % os.getpid())

            ##############
            # Early stop #
            ##############
            unchanged += 1
            if unchanged > 15:
                say('\n\nEARLY STOP\n')
                break

            say('\n\n\nEpoch: %d' % (epoch + 1))
            say('\n  TRAIN\n  ')

            np.random.shuffle(cand_indices)

            ############
            # Training #
            ############
            start = time.time()
            for index, b_index in enumerate(cand_indices):
                if index != 0 and index % 100 == 0:
                    say("  {}/{}".format(index, len(cand_indices)))

                cost, g_norm, pred_a, pred_r = train_f(b_index)
                binned_n_agents, labels_a, labels_r = evalset[b_index]

                if math.isnan(cost):
                    say('\n\nLoss is NAN: Mini-Batch Index: %d\n' % index)
                    exit()

                evaluator.update(binned_n_agents, cost, g_norm, pred_a, pred_r, labels_a, labels_r)

            end = time.time()
            say('\n\tTime: %f' % (end - start))
            say("\n\tp_norm: {}\n".format(self.get_pnorm_stat()))
            evaluator.show_results()

            ##############
            # Validating #
            ##############
            if argv.dev_data:
                say('\n\n  DEV\n  ')
                dev_acc_both = self.evaluate_all(dev_samples, pred_f, pred_r_f)

                if dev_acc_both > best_dev_acc_both:
                    unchanged = 0
                    best_dev_acc_both = evaluator.acc_both
                    acc_history[epoch+1] = [best_dev_acc_both]

                    if argv.save:
                        fn = 'Model-%s.unit-%s.attention-%d.batch-%d.reg-%f.sents-%d.words-%d' %\
                             (argv.model, argv.unit, argv.attention, argv.batch, argv.reg, n_prev_sents, max_n_words)
                        dump_data(self, fn)

            if argv.test_data:
                say('\n\n\r  TEST\n  ')
                test_acc_both = self.evaluate_all(test_samples, pred_f, pred_r_f)

                if unchanged == 0:
                    if epoch+1 in acc_history:
                        acc_history[epoch+1].append(test_acc_both)
                    else:
                        acc_history[epoch+1] = [test_acc_both]

            #####################
            # Show best results #
            #####################
            say('\n\n\tBEST ACCURACY HISTORY')
            for k, v in sorted(acc_history.items()):
                if len(v) == 2:
                    say('\n\tEPOCH-{:d}  \tBEST DEV ACC:{:.2%}\tBEST TEST ACC:{:.2%}'.format(k, v[0], v[1]))
                else:
                    say('\n\tEPOCH-{:d}  \tBEST DEV ACC:{:.2%}'.format(k, v[0]))


class DynamicModel(Model):

    def __init__(self, argv, max_n_agents, n_vocab, init_emb):
        super(DynamicModel, self).__init__(argv, max_n_agents, n_vocab, init_emb)

    def compile(self, c, r, a, y_r, y_a, n_agents):
        self.inputs = [c, r, a, y_r, y_a, n_agents]
        batch = T.cast(c.shape[0], 'int32')
        n_prev_sents = T.cast(c.shape[1], 'int32')

        ################
        # Architecture #
        ################
        x_c, x_r = self.lookup_table(c, r, batch, n_prev_sents)
        h_c, h_r = self.intra_sentential_layer(x_c, x_r, batch, n_prev_sents)
        a_res, A_adr, z = self.inter_sentential_layer(h_c, a, n_agents, batch)
        p_a, p_r = self.output_layer(a_res, A_adr, z, h_r)

        ######################
        # Training & Testing #
        ######################
        self.nll, self.cost = self.objective_f(p_a, p_r, y_a, y_r, n_agents)
        self.optimization()
        self.predict(p_a, p_r)

    def lookup_table(self, c, r, batch, n_prev_sents):
        pad = build_shared_zeros((1, self.dim_emb))
        if self.init_emb is None:
            emb = theano.shared(initialize_weights(self.n_vocab - 1, self.dim_emb))
            self.params += [emb]
        else:
            emb = theano.shared(self.init_emb)

        E = T.concatenate([pad, emb], 0)

        x_c = E[c]
        x_r = E[r]

        x_c = x_c.reshape((batch * n_prev_sents, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)
        x_r = x_r.reshape((batch * self.n_cands, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)

        return x_c, x_r

    def intra_sentential_layer(self, x_c, x_r, batch, n_prev_sents):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            encoder = rnn.RNN(n_i=self.dim_emb, n_h=self.dim_hidden)
        else:
            encoder = rnn.GRU(n_i=self.dim_emb, n_h=self.dim_hidden)
        self.params += encoder.params

        ##########
        # Encode #
        ##########
        h_c0 = T.zeros(shape=(batch * n_prev_sents, self.dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * self.n_cands, self.dim_hidden), dtype=theano.config.floatX)

        # 1D: n_words, 2D: batch * n_prev_sents, 3D: n_hidden
        H_c, _ = theano.scan(fn=encoder.recurrence, sequences=x_c, outputs_info=h_c0)
        h_c = H_c[-1].reshape((batch, n_prev_sents, self.dim_hidden)).dimshuffle(1, 0, 2)

        # 1D: n_words, 2D: batch * n_cands, 3D: n_hidden
        H_r, _ = theano.scan(fn=encoder.recurrence, sequences=x_r, outputs_info=h_r0)
        h_r = H_r[-1].reshape((batch, self.n_cands, self.dim_hidden))

        return h_c, h_r

    def inter_sentential_layer(self, h_c, a, n_agents, batch):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            encoder = rnn.RNN(n_i=self.dim_hidden, n_h=self.dim_hidden)
        else:
            encoder = rnn.GRU(n_i=self.dim_hidden, n_h=self.dim_hidden)
        self.params += encoder.params

        a = a[:, :, :n_agents].dimshuffle(1, 0, 2)

        # 1D: n_prev_sents, 2D: batch, 3D: n_agents, 4D: dim_agent
        A0 = T.zeros(shape=(batch, n_agents, self.dim_hidden), dtype=theano.config.floatX)
        A, _ = theano.scan(fn=encoder.recurrence_inter, sequences=[h_c, a], outputs_info=A0)

        ###########################
        # Extract summary vectors #
        ###########################
        # Responding Agent: 1D: Batch, 2D: dim_agent
        a_res = A[-1][:, 0]

        # Addressee Agents: 1D: batch, 2D: n_agents - 1, 3D: dim_agent
        A_adr = A[-1][:, 1:]

        #############
        # Attention #
        #############
        # 1D: batch, 2D: dim_agent
        if self.attention:
            encoder = attention.Attention(n_h=self.dim_hidden, attention=self.attention)
            self.params += encoder.params
            z = encoder.f(A[-1], a_res)
        else:
            z = T.max(A[-1], 1)

        return a_res, A_adr, z

    def output_layer(self, a_res, A_adr, z, h_r):
        W_oa = theano.shared(initialize_weights(self.dim_hidden * 2, self.dim_hidden))
        W_or = theano.shared(initialize_weights(self.dim_hidden * 2, self.dim_hidden))
        self.params += [W_oa, W_or]

        o = T.concatenate([a_res, z], axis=1)  # 1D: batch, 2D: dim_hidden + dim_hidden
        o_a = T.dot(o, W_oa)  # 1D: batch, 2D: dim_hidden
        o_r = T.dot(o, W_or)  # 1D: batch, 2D: dim_hidden

        score_a = T.batched_dot(o_a, A_adr.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_agents-1; elem=scalar
        score_r = T.batched_dot(o_r, h_r.dimshuffle(0, 2, 1))  # 1D: batch, 2D: n_cands; elem=scalar

        p_a = sigmoid(score_a)  # 1D: batch, 2D: n_agents-1; elem=scalar
        p_r = sigmoid(score_r)  # 1D: batch, 2D: n_cands; elem=scalar

        return p_a, p_r


class StaticModel(Model):

    def __init__(self, argv, max_n_agents, n_vocab, init_emb):
        super(StaticModel, self).__init__(argv, max_n_agents, n_vocab, init_emb)

    def compile(self, c, r, a, y_r, y_a, n_agents):
        self.inputs = [c, r, a, y_r, y_a, n_agents]
        batch = T.cast(c.shape[0], 'int32')
        n_prev_sents = T.cast(c.shape[1], 'int32')

        ################
        # Architecture #
        ################
        x_c, x_r = self.lookup_table(c, r)
        a_res, h_c, h_r, A_p = self.intra_sentential_layer(a, n_agents, x_c, x_r, batch, n_prev_sents)
        p_a, p_r = self.output_layer(a_res, h_c, h_r, A_p)

        ######################
        # Training & Testing #
        ######################
        self.nll, self.cost = self.objective_f(p_a, p_r, y_a, y_r, n_agents)
        self.optimization()
        self.predict(p_a, p_r)

    def lookup_table(self, c, r):
        pad = build_shared_zeros((1, self.dim_emb))
        if self.init_emb is None:
            emb = theano.shared(initialize_weights(self.n_vocab - 1, self.dim_emb))
            self.params += [emb]
        else:
            emb = theano.shared(self.init_emb)

        E = T.concatenate([pad, emb], 0)

        x_c = E[c]
        x_r = E[r]

        return x_c, x_r

    def intra_sentential_layer(self, a, n_agents, x_c, x_r, batch, n_prev_sents):
        ############
        # Encoders #
        ############
        if self.unit == 'rnn':
            encoder = rnn.RNN(n_i=self.dim_emb, n_h=self.dim_hidden)
        else:
            encoder = rnn.GRU(n_i=self.dim_emb, n_h=self.dim_hidden)
        self.params += encoder.params

        A = theano.shared(initialize_weights(self.max_n_agents, self.dim_emb))
        self.params += [A]

        ##########
        # Encode #
        ##########
        h_c0 = T.zeros(shape=(batch, self.dim_hidden), dtype=theano.config.floatX)
        h_r0 = T.zeros(shape=(batch * self.n_cands, self.dim_hidden), dtype=theano.config.floatX)

        x_a = T.dot(a, A).reshape((batch, n_prev_sents, 1, self.dim_emb))
        x_c = T.concatenate([x_a, x_c], 2)
        x_c = x_c.reshape((batch,  n_prev_sents * (self.n_words + 1), self.dim_emb)).dimshuffle(1, 0, 2)
        x_r = x_r.reshape((batch * self.n_cands, self.n_words, self.dim_emb)).dimshuffle(1, 0, 2)

        # H_c: 1D: c_words, 2D: batch, 3D: dim_hidden
        # H_r: 1D: r_words, 2D: batch * n_cands, 3D: dim_hidden
        H_c, _ = theano.scan(fn=encoder.recurrence, sequences=x_c, outputs_info=h_c0)
        H_r, _ = theano.scan(fn=encoder.recurrence, sequences=x_r, outputs_info=h_r0)

        ###########################
        # Extract summary vectors #
        ###########################
        a_res = T.repeat(A[0].dimshuffle('x', 0), batch, 0)
        h_c = H_c[-1]
        h_r = H_r[-1].reshape((batch, self.n_cands, self.dim_hidden))
        A_p = A[1: n_agents]  # 1D: batch, 2D: n_agents - 1, 3D: dim_emb

        return a_res, h_c, h_r, A_p

    def output_layer(self, a_res, h_c, h_r, A_p):
        W_oa = theano.shared(initialize_weights(self.dim_emb + self.dim_hidden, self.dim_emb))
        W_or = theano.shared(initialize_weights(self.dim_emb + self.dim_hidden, self.dim_hidden))
        self.params += [W_oa, W_or]

        o = T.concatenate([a_res, h_c], 1)  # 1D: batch, 2D: dim_emb + dim_hidden
        o_a = T.dot(o, W_oa)  # 1D: batch, 2D: dim_hidden
        o_r = T.dot(o, W_or)  # 1D: batch, 2D: dim_hidden

        score_a = T.dot(o_a, A_p.T)
        score_r = T.batched_dot(o_r, h_r.dimshuffle(0, 2, 1))

        p_a = sigmoid(score_a)  # 1D: batch, 2D: n_agents-1; elem=scalar
        p_r = sigmoid(score_r)  # 1D: batch, 2D: n_cands; elem=scalar

        return p_a, p_r

