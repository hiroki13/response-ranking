import math
import time

import numpy as np
import theano
import theano.tensor as T

from model import StaticModel, DynamicModel
from ..utils import say, load_data, dump_data, create_path, move_data, check_identifier
from ..utils.evaluator import Evaluator


class ModelAPI(object):
    def __init__(self, argv, init_emb, vocab, n_prev_sents):
        self.argv = argv
        self.init_emb = init_emb
        self.vocab = vocab
        self.max_n_agents = n_prev_sents + 1

        self.model = None
        self.train_f = None
        self.pred_f = None
        self.pred_r_f = None

    def set_model(self):
        say('\n\nBUILD A MODEL\n')
        argv = self.argv

        #####################
        # Network variables #
        #####################
        c = T.itensor3('c')
        r = T.itensor3('r')
        a = T.ftensor3('a')
        y_r = T.ivector('y_r')
        y_a = T.imatrix('y_a')
        n_agents = T.iscalar('n_agents')

        max_n_agents = self.max_n_agents
        init_emb = self.init_emb
        n_vocab = self.vocab.size()

        #################
        # Build a model #
        #################
        say('MODEL: %s  Unit: %s  Opt: %s  Activation: %s  ' % (argv.model, argv.unit, argv.opt, argv.activation))

        if argv.model == 'static':
            model = StaticModel
        else:
            model = DynamicModel

        self.model = model(argv, max_n_agents, n_vocab, init_emb)
        self.model.compile(c=c, r=r, a=a, y_r=y_r, y_a=y_a, n_agents=n_agents)

    def save_params(self, fn=None, dir='../data/model'):
        argv = self.argv
        if fn is None:
            fn = 'param.model-%s.unit-%s.hidden-%d' % (argv.model, argv.unit, argv.dim_hidden)
        fn = check_identifier(fn)
        dump_data(data=map(lambda p: p.get_value(borrow=True), self.model.params), fn=fn)
        create_path(dir)
        move_data(fn, dir)

    def load_params(self):
        params = load_data(self.argv.load_param)
        assert len(self.model.params) == len(params)
        for p1, p2 in zip(self.model.params, params):
            p1.set_value(p2)

    def set_train_f(self, train_samples):
        model = self.model
        index = T.iscalar('index')
        self.train_f = theano.function(inputs=[index],
                                       outputs=[model.nll, model.g_norm, model.a_hat, model.r_hat],
                                       updates=model.update,
                                       givens={
                                           model.inputs[0]: train_samples[0][index],
                                           model.inputs[1]: train_samples[1][index],
                                           model.inputs[2]: train_samples[2][index],
                                           model.inputs[3]: train_samples[3][index],
                                           model.inputs[4]: train_samples[4][index],
                                           model.inputs[5]: train_samples[5][index]
                                       }
                                       )

    def set_train_online_f(self):
        model = self.model
        self.train_f = theano.function(inputs=model.inputs,
                                       outputs=[model.nll, model.g_norm, model.a_hat, model.r_hat],
                                       updates=model.update,
                                       )

    def set_test_f(self):
        model = self.model
        self.pred_f = theano.function(inputs=model.inputs,
                                      outputs=[model.a_hat, model.r_hat],
                                      on_unused_input='ignore'
                                      )
        self.pred_r_f = theano.function(inputs=model.inputs,
                                        outputs=model.r_hat,
                                        on_unused_input='ignore'
                                        )

    def train(self, c, r, a, res_vec, adr_vec, n_agents):
        nll, g_norm, pred_a, pred_r = self.train_f(c, r, a, res_vec, adr_vec, n_agents)
        return nll, g_norm, pred_a, pred_r

    def train_all(self, batch_indices, evalset):
        evaluator = Evaluator()
        np.random.shuffle(batch_indices)
        start = time.time()

        for index, b_index in enumerate(batch_indices):
            if index != 0 and index % 100 == 0:
                say("  {}/{}".format(index, len(batch_indices)))

            cost, g_norm, pred_a, pred_r = self.train_f(b_index)
            binned_n_agents, labels_a, labels_r = evalset[b_index]

            if math.isnan(cost):
                say('\n\nLoss is NAN: Mini-Batch Index: %d\n' % index)
                exit()

            evaluator.update(binned_n_agents, cost, g_norm, pred_a, pred_r, labels_a, labels_r)

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        evaluator.show_results()

    def predict(self, c, r, a, y_r, y_a, n_agents):
        pred_a = None
        if n_agents > 1:
            pred_a, pred_r = self.pred_f(c, r, a, y_r, y_a, n_agents)
        else:
            pred_r = self.pred_r_f(c, r, a, y_r, y_a, n_agents)
        return pred_a, pred_r

    def predict_all(self, samples):
        evaluator = Evaluator()
        start = time.time()

        for i, sample in enumerate(samples):
            if i != 0 and i % 100 == 0:
                say("  {}/{}".format(i, len(samples)))

            x = sample[0]
            binned_n_agents = sample[1]
            labels_a = sample[2]
            labels_r = sample[3]
            pred_a, pred_r = self.predict(c=x[0], r=x[1], a=x[2], y_r=x[3], y_a=x[4], n_agents=x[5])

            evaluator.update(binned_n_agents, 0., 0., pred_a, pred_r, labels_a, labels_r)

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        evaluator.show_results()

        return evaluator.acc_both, evaluator.acc_adr, evaluator.acc_res

    def get_pnorm_stat(self):
        lst_norms = []
        for p in self.model.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    """
    def output(self, fn, dataset, vocab):
        f = gzip.open(fn + '.gz', 'wb')
        f.writelines('RESULTS')
        index = 0

        for i in xrange(len(self.answer_r)):
            answer_a = self.answer_a[i]
            answer_r = self.answer_r[i]

            for j in xrange(len(answer_r)):
                s = dataset[index]
                agent_ids = Vocab()
                agent_ids.add_word(s.speaker_id)
                index += 1

                f.writelines('\n\n\n%d\nCONTEXT\n' % index)
                for c in s.orig_context:
                    f.writelines('%s\t%s\t%s\t' % (c[0], c[1], c[2]))

                    for w in c[-1]:
                        f.writelines(vocab.get_word(w) + ' ')
                    f.writelines('\n')

                for c in reversed(s.orig_context):
                    agent_ids.add_word(c[1])

                f.writelines('\nRESPONSE\n')
                f.writelines('%s\t%s\t%s\n' % (s.time, s.speaker_id, s.addressee_id))
                for k, r in enumerate(s.responses):
                    f.writelines('\t(%d) ' % k)
                    for w in r:
                        f.writelines(vocab.get_word(w) + ' ')
                    f.writelines('\n')
                f.writelines('\nPREDICTION\n')
                if answer_a is None:
                    f.writelines('Gold ADR: %s\tPred ADR: NONE\tGold RES: %d\tPred RES: %d' %
                                 (s.addressee_id, s.label, answer_r[j]))
                else:
                    f.writelines('Gold ADR: %s\tPred ADR: %s\tGold RES: %d\tPred RES: %d' %
                                 (s.addressee_id, agent_ids.get_word(answer_a[j]+1), s.label, answer_r[j]))
        f.close()
        """

