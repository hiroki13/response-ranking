import numpy as np

from ..utils import say


class Evaluator(object):

    def __init__(self):
        self.index = 0.
        self.ttl_cost = 0.
        self.ttl_g_norm = 0.

        self.crr_adr = 0
        self.crr_res = 0
        self.crr_both = 0
        self.total = 0.

        self.acc_adr = 0.
        self.acc_res = 0.
        self.acc_both = 0.

        # 1D: binned_n_agents, 2D: (crr_a, crr_r, crr_both, total)
        self.results = np.zeros((7, 4), dtype='float32')  # 2, 3, 4, 5, 6-10, 11-

    def update(self, n_agents, cost, g_norm, pred_a, pred_r, gold_a, gold_r):
        self.index += 1

        if pred_a is not None:
            crr_adr_vec = map(lambda x: 1 if x[0] == x[1] else 0, zip(pred_a, gold_a))
        else:
            crr_adr_vec = []

        crr_res_vec = map(lambda x: 1 if x[0] == x[1] else 0, zip(pred_r, gold_r))
        crr_adr = np.sum(crr_adr_vec)
        crr_res = np.sum(crr_res_vec)

        for i, a in enumerate(crr_adr_vec):
            n = n_agents[i]
            self.results[n][0] += a

        for i, r in enumerate(crr_res_vec):
            n = n_agents[i]
            self.results[n][1] += r
            self.results[n][3] += 1

        crr_both = 0.
        if pred_a is not None and pred_r is not None:
            for a, r, n in zip(crr_adr_vec, crr_res_vec, n_agents):
                crr_both += a * r
                self.results[n][2] += a * r

        self.crr_adr += crr_adr
        self.crr_res += crr_res
        self.crr_both += crr_both
        self.total += len(pred_r)

        self.ttl_cost += cost
        self.ttl_g_norm += g_norm

    def show_results(self):
        acc_adr = self.acc_adr = self.crr_adr / self.total
        acc_res = self.acc_res = self.crr_res / self.total
        acc_both = self.acc_both = self.crr_both / self.total

        say('\n\tTotal NLL: %f\tTotal Grad Norm: %f' % (self.ttl_cost, self.ttl_g_norm))
        say('\n\tAvg.  NLL: %f\tAvg.  Grad Norm: %f' % (self.ttl_cost / self.index, self.ttl_g_norm / self.index))

        say('\n\n\tAccuracy\n\t\tBoth: %f (%d/%d)  Addressee: %f (%d/%d)  Response: %f (%d/%d)' %
            (acc_both, self.crr_both, self.total, acc_adr, self.crr_adr, self.total, acc_res, self.crr_res, self.total))

        say('\n\n\tAccuracy for each Binned Num of Agents in Context')
        for n_agents in xrange(self.results.shape[0]):
            result = self.results[n_agents]
            crr_adr = result[0]
            crr_res = result[1]
            crr_both = result[2]
            total = result[3]

            acc_adr = crr_adr / total
            acc_res = crr_res / total
            acc_both = crr_both / total

            say('\n\t  %d\tBoth: %f (%d/%d)  Addressee: %f (%d/%d)  Response: %f (%d/%d)' %
                (n_agents+1, acc_both, crr_both, total, acc_adr, crr_adr, total, acc_res, crr_res, total))

