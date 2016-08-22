import math
import gzip

import theano
import theano.tensor as T

from ling.vocab import Vocab


class Decoder(object):
    def __init__(self, argv, model):
        self.argv = argv
        self.max_n_agents = argv.n_prev_sents + 1

        self.model = model
        self.train_f = None
        self.pred_f = None
        self.pred_r_f = None

        self.answer_a = []
        self.answer_r = []
        self.prob_r = []

    def set_train_f(self, tr_samples):
        index = T.iscalar('index')
        self.train_f = theano.function(inputs=[index],
                                       outputs=[self.model.nll, self.model.correct_a, self.model.correct_r],
                                       updates=self.model.update,
                                       givens={
                                           self.model.tr_inputs[0]: tr_samples[0][index],
                                           self.model.tr_inputs[1]: tr_samples[1][index],
                                           self.model.tr_inputs[2]: tr_samples[2][index],
                                           self.model.tr_inputs[3]: tr_samples[3][index],
                                           self.model.tr_inputs[4]: tr_samples[4][index],
                                           self.model.tr_inputs[5]: tr_samples[5][index],
                                           self.model.tr_inputs[6]: tr_samples[6][index],
                                           self.model.tr_inputs[7]: tr_samples[7][index]
                                       }
                                       )

    def set_test_f(self):
        self.pred_f = theano.function(inputs=self.model.pr_inputs,
                                      outputs=[self.model.correct_a, self.model.correct_r,
                                               self.model.y_a_hat, self.model.y_r_hat]
                                      )
        if self.argv.model == 'static':
            self.pred_r_f = theano.function(inputs=self.model.pr_inputs[:-2],
                                            outputs=[self.model.correct_r, self.model.y_r_hat]
                                            )
        else:
            self.pred_r_f = theano.function(inputs=self.model.pr_inputs[:-1],
                                            outputs=[self.model.correct_r, self.model.y_r_hat]
                                            )

    def train(self, index):
        avg_cost, pred_a, pred_r = self.train_f(index)
        assert not math.isnan(avg_cost), 'Mini-Batch Index: %d' % index
        assert len(pred_a) == len(pred_r)
        return avg_cost, pred_a, pred_r

    def predict(self, sample):
        c = sample[0]
        r = sample[1]
        a = sample[2]
        y_r = sample[3]
        y_a = sample[5]
        n_agents = sample[-1]

        pred_a = None
        answer_a = None

        if n_agents > 1:
            pred_a, pred_r, answer_a, answer_r = self.pred_f(c, r, a, y_r, n_agents, y_a)
            assert len(pred_a) == len(pred_r)
        else:
            if self.argv.model == 'static':
                pred_r, answer_r = self.pred_r_f(c, r, a, y_r)
            else:
                pred_r, answer_r = self.pred_r_f(c, r, a, y_r, n_agents)

        self.answer_a.append(answer_a)
        self.answer_r.append(answer_r)
        return pred_a, pred_r

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

