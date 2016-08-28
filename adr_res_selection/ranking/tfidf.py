import sys
from collections import Counter

from ..utils import say, load_dataset
from ..ling import Vocab
from preprocessor import get_samples

import numpy as np


class Model(object):
    def __init__(self, docs):
        """
        :param docs: 1D: n_samples; 2D: Sample
        """

        # 1D: n_samples; elem=Sample
        self.doc_c, self.doc_r = self.get_docs(docs)

        self.doc_c_set = self.get_set_docs(self.doc_c)
        self.doc_r_set = self.get_set_docs(self.doc_r)

        self.tf_c = None
        self.tf_r = None

        # TODO
        self.docs_set = self.get_set_docs()

        self.vocab = Counter(self.get_all_words())
        self.vocab_ids = self.vocab.keys()
        self.word_freq_in_doc = Counter()

        self.n_docs = float(len(docs))
        self.n_word_types = len(self.vocab.values())  # num of types

        self.idf = self.get_idf()

    def get_docs(self, docs):
        doc_c = []
        doc_r = []
        for sample in docs:
            doc_c.append(reduce(lambda x, y: x + y, sample.context, []))
            doc_r.append(sample.responses[sample.label])
        return doc_c, doc_r

    def get_set_docs(self, docs):
        return [set(doc) for doc in docs]

    def get_all_words(self):
        return reduce(lambda x, y: x + y, self.docs, [])

    def get_n_words_in_doc(self, doc):
        self.word_freq_in_doc.clear()
        self.word_freq_in_doc.update(doc)
        return self.word_freq_in_doc, float(len(doc))

    def get_tf(self, word_freq_in_doc, n_words_in_doc):
        return word_freq_in_doc / n_words_in_doc

    def get_idf(self):
        say('\nSet IDF...\n')
        df = Counter(reduce(lambda x, y: x + list(y), self.docs_set, []))
        df = np.asarray([value for key, value in sorted(df.items(), key=lambda x: x[0])], dtype='float32')
        idf = np.log(self.n_docs / df)
        return idf

    def get_tfidf(self, tf, idf):
        return tf * idf

    def get_tfidf_vec(self, doc):
        v = [0. for i in xrange(self.n_word_types)]
        word_freq_in_doc, n_words_in_doc = self.get_n_words_in_doc(doc)

        for word_id in doc:
            tf = self.get_tf(word_freq_in_doc=word_freq_in_doc[word_id], n_words_in_doc=n_words_in_doc)
            idf = self.idf[word_id]
            v[word_id] = self.get_tfidf(tf, idf)

        return np.asarray(v, dtype='float32')

    def compute(self, dataset):
        print 'TF-IDF COMPUTING START'
        crr_a = 0.
        crr_r = 0.
        crr_b = 0.
        results_a = [0. for i in xrange(7)]
        results_r = [0. for i in xrange(7)]
        results_b = [0. for i in xrange(7)]
        total = [0. for i in xrange(7)]

        for i, sample in enumerate(dataset):
            if i != 0 and i % 1000 == 0:
                print '%d:%f' % (i, crr_r/i)
                sys.stdout.flush()

            adr = get_addressee(sample)
            c_vec = self.get_tfidf_vec(set_sent(sample.context))
            best_r = -1
            best_score = -100000000000000.0
            for j, r in enumerate(sample.response):
                r_vec = self.get_tfidf_vec(r)
                score = np.dot(c_vec, r_vec)
                if score > best_score:
                    best_r = j
                    best_score = score

            ans_a = 1 if adr == sample.true_addressee else 0
            ans_r = 1 if best_r == sample.true_response else 0
            ans_b = 1 if ans_a == ans_r == 1 else 0
            crr_a += ans_a
            crr_r += ans_r
            crr_b += ans_b
            results_a[sample.n_agents_in_ctx] += ans_a
            results_r[sample.n_agents_in_ctx] += ans_r
            results_b[sample.n_agents_in_ctx] += ans_b
            total[sample.n_agents_in_ctx] += 1
        print 'Accuracy  Both:%f Adr:%f Res:%f' % (crr_b / len(dataset), crr_a / len(dataset), crr_r / len(dataset))

        for n in xrange(len(total)):
            print '%d: BOTH  Accuracy:%f Total:%d Correct:%d' % (n, results_b[n]/total[n], total[n], results_b[n])
            print '%d: ADR   Accuracy:%f Total:%d Correct:%d' % (n, results_a[n]/total[n], total[n], results_a[n])
            print '%d: RES   Accuracy:%f Total:%d Correct:%d' % (n, results_r[n]/total[n], total[n], results_r[n])
            print


def get_addressee(sample):
    t = len(sample.context) - 1
    while t > -1:
        agent_id = sample.speaking_agent_one_hot_vector[t]
        if agent_id != 0:
            return agent_id
        t -= 1


def set_sent(sent):
    return reduce(lambda x, y: x + y, sent, [])


def main(argv):
    print 'SET UP TRAINING SETTINGS'

    ###########################
    # Task setting parameters #
    ###########################
    n_cands = argv.n_cands
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    say('\n\nTASK  SETTING')
    say('\n  Response Candidates:%d  Contexts:%d  Max Word Num:%d' % (n_cands, n_prev_sents, max_n_words))

    #################
    # Load datasets #
    #################
    # data: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\n\nLoad data sets...')
    train_data, vocab_word = load_dataset(argv.train_data, argv.check)
    dev_data, vocab_word = load_dataset(argv.dev_data, argv.check, vocab_word)
    test_data, vocab_word = load_dataset(argv.test_data, argv.check, vocab_word)

    #############
    # Set vocab #
    #############
    token_dict = Vocab()
    for w in vocab_word:
        token_dict.add_word(w)

    ################################
    # Set up experimental settings #
    ################################
    # dataset: 1D: n_samples; 2D: Sample
    say('\n\nSetting datasets...')

    train_dataset = get_samples(threads=train_data, vocab=token_dict.w2i, n_prev_sents=n_prev_sents,
                                max_n_words=max_n_words, test=False, sample_size=argv.sample_size)
    say('\n\nTRAIN SETTING\tVocab:%d' % token_dict.size())

    if argv.dev_data:
        dev_dataset = get_samples(threads=dev_data, vocab=token_dict.w2i, n_prev_sents=n_prev_sents,
                                  max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        train_dataset += dev_dataset

    if argv.test_data:
        test_dataset = get_samples(threads=test_data, vocab=token_dict.w2i, n_prev_sents=n_prev_sents,
                                   max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        train_dataset += test_dataset

    model = Model(docs=train_dataset)
    say('\nThreads: %d  Words: %d\n' % (model.n_docs, model.n_word_types))

    model.compute(dev_dataset)
