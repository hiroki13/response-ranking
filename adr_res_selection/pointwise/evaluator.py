import numpy as np
from utils.io_utils import say


class Evaluator(object):

    def __init__(self):
        self.ttl_cost = 0.
        self.crr_a = 0
        self.crr_r = 0
        self.crr_t = 0
        self.total = 0.
        self.acc_a = 0.
        self.acc_r = 0.
        self.acc_t = 0.

        self.results = np.zeros((7, 4), dtype='float32')  # 2, 3, 4, 5, 6-10, 11-

    def update(self, n_agents, ttl_cost, pred_a, pred_r):
        crr_a = np.sum(pred_a) if pred_a is not None else 0
        crr_r = np.sum(pred_r) if pred_r is not None else 0
        crr_ar = 0.
        ttl_sample = len(pred_r)

        if pred_a is not None:
            for i in xrange(len(pred_a)):
                self.results[n_agents[i]][0] += pred_a[i]

        if pred_r is not None:
            for i in xrange(len(pred_r)):
                n = n_agents[i]
                self.results[n][1] += pred_r[i]
                self.results[n][3] += 1

        if pred_a is not None and pred_r is not None:
            for pa, pr, n in zip(pred_a, pred_r, n_agents):
                if pa == pr == 1:
                    crr_ar += 1
                    self.results[n][2] += 1

        self.ttl_cost += ttl_cost
        self.crr_a += crr_a
        self.crr_r += crr_r
        self.crr_t += crr_ar
        self.total += ttl_sample

    def show_results(self):
        self.acc_a = self.crr_a / self.total
        self.acc_r = self.crr_r / self.total
        self.acc_t = self.crr_t / self.total

        say('\n\tTotal NLL: %f' % self.ttl_cost)
        say('\n\n\tAccuracy\n\t\tBoth: %f (%d/%d)  Addressee: %f (%d/%d)  Response: %f (%d/%d)' %
            (self.acc_t, self.crr_t, self.total, self.acc_a, self.crr_a, self.total, self.acc_r, self.crr_r, self.total))

        say('\n\n\tAccuracy for each Binned Num of Agents in Context')
        for n_agents in xrange(self.results.shape[0]):
            result = self.results[n_agents]
            crr_a = result[0]
            crr_r = result[1]
            crr_t = result[2]
            total = result[3]

            acc_a = crr_a / total
            acc_r = crr_r / total
            acc_t = crr_t / total

            say('\n\t  %d\tBoth: %f (%d/%d)  Addressee: %f (%d/%d)  Response: %f (%d/%d)' %
                (n_agents+1, acc_t, crr_t, total, acc_a, crr_a, total, acc_r, crr_r, total))

        self.ttl_cost = 0.
        self.crr_a = 0
        self.crr_r = 0
        self.crr_t = 0
        self.total = 0.
        self.results = np.zeros((7, 4), dtype='float32')

