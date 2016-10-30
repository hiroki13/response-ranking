import sys
from collections import Counter

from ..utils import say, load_dataset
from ..utils.evaluator import Evaluator
from preprocessor import get_samples, convert_sample_into_ids

import numpy as np


class Model(object):
    def __init__(self, docs):
        """
        :param docs: 1D: n_samples; 2D: Sample
        """

        # 1D: n_samples; elem=Sample
        say('Setting Documents...\n')
        self.doc = self.get_docs(docs)
        say('Setting Document sets...\n')
        self.doc_set = self.get_set_docs(self.doc)

        say('Setting Vocab...\n')
        self.vocab = Counter(self.get_all_words())
        self.vocab_ids = self.vocab.keys()
        self.word_freq_in_doc = Counter()

        self.n_docs = float(len(docs))
        self.n_word_types = len(self.vocab.values())  # num of types

        say('Setting IDF...\n')
        self.idf = self.get_idf()

        self.tf_c = None
        self.tf_r = None

    def get_docs(self, docs):
        d = []
        for sample in docs:
            r = sample.response[sample.true_res]
            c = []
            for ctx in sample.context:
                c += ctx
            d.append(c + r)
        return d

    def get_set_docs(self, docs):
        return [set(doc) for doc in docs]

    def get_all_words(self):
        words = []
        for d in self.doc:
            words += d
        return words

    def get_n_words_in_doc(self, doc):
        self.word_freq_in_doc.clear()
        self.word_freq_in_doc.update(doc)
        return self.word_freq_in_doc, float(len(doc))

    def get_tf(self, word_freq_in_doc, n_words_in_doc):
        return word_freq_in_doc / n_words_in_doc

    def get_df(self):
        doc = []
        for d in self.doc_set:
            doc += list(d)
        return Counter(doc)

    def get_idf(self):
        df = self.get_df()
        df = np.asarray([value for key, value in sorted(df.items(), key=lambda x: x[0])], dtype='float32')
        idf = np.log(self.n_docs / df)
        say('\nIDF Table: %d\n' % len(idf))
        return idf

    def get_tfidf(self, tf, idf):
        return tf * idf

    def get_tfidf_vec(self, doc):
        v = [0. for i in xrange(self.n_word_types)]
        word_freq_in_doc, n_words_in_doc = self.get_n_words_in_doc(doc)

        for word_id in set(doc):
            if word_id < 0:
                continue
            tf = self.get_tf(word_freq_in_doc=word_freq_in_doc[word_id], n_words_in_doc=n_words_in_doc)
            idf = self.idf[word_id]
            v[word_id] = self.get_tfidf(tf, idf)

        return np.asarray(v, dtype='float32')

    def get_best_response(self, response, c_vec):
        best_r = -1
        best_score = -100000000000000.0
        for j, r in enumerate(response):
            r_vec = self.get_tfidf_vec(r)
            score = np.dot(c_vec, r_vec)
            if score > best_score:
                best_r = j
                best_score = score
        return best_r

    def compute(self, samples):
        say('\nTF-IDF COMPUTING START\n')
        evaluator = Evaluator()

        for i, sample in enumerate(samples):
            if i != 0 and i % 1000 == 0:
                print '%d ' % i,
                sys.stdout.flush()

            pred_adr = get_adr_index(sample)
            c_vec = self.get_tfidf_vec(set_sent(sample.context))
            pred_res = self.get_best_response(sample.response, c_vec)

            evaluator.update([sample.binned_n_agents_in_ctx], 0., 0.,
                             [pred_adr], [pred_res], [sample.true_adr], [sample.true_res])

        evaluator.show_results()


def get_adr_index(sample):
    """
    Responding agent index = 0
    Candidate adr agent index = 1 - N
    """
    t = len(sample.context) - 1
    while t > -1:
        agent_index = sample.spk_agents[t]
        if agent_index > 0:
            return agent_index - 1
        t -= 1
    return -1


def set_sent(sent):
    s = []
    for c in sent:
        s.extend(c)
    return s


def get_datasets(argv):
    data_size = argv.data_size

    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\n\nLoad dataset...')
    train_dataset, word_set = load_dataset(fn=argv.train_data, data_size=data_size)
    dev_dataset, _ = load_dataset(fn=argv.dev_data, data_size=data_size)
    test_dataset, _ = load_dataset(fn=argv.test_data, data_size=data_size)

    return train_dataset, dev_dataset, test_dataset, word_set


def create_samples(argv, train_dataset, dev_dataset, test_dataset):
    n_prev_sents = argv.n_prev_sents
    sample_size = argv.sample_size

    # samples: 1D: n_samples; elem=Sample()
    say('\n\nCreating samples...')
    train_samples = get_samples(threads=train_dataset, n_prev_sents=n_prev_sents)
    dev_samples = get_samples(threads=dev_dataset, n_prev_sents=n_prev_sents, test=True)
    test_samples = get_samples(threads=test_dataset, n_prev_sents=n_prev_sents, test=True)

    ##########################
    # Limit the used samples #
    ##########################
    if sample_size > 1:
        np.random.shuffle(train_samples)
        train_samples = train_samples[: (len(train_samples) / sample_size)]

    return train_samples, dev_samples, test_samples


def convert_samples(train_samples, dev_samples, test_samples):
    say('\n\nConverting words into ids...')
    train_samples, vocab_word = convert_sample_into_ids(train_samples)
    dev_samples, _ = convert_sample_into_ids(dev_samples, vocab_word)
    test_samples, _ = convert_sample_into_ids(test_samples, vocab_word)
    say('\n\tVocab size: %d' % vocab_word.size())

    return train_samples, dev_samples, test_samples, vocab_word


def main(argv):
    say('\nSET UP TRAINING SETTINGS\n')

    ##############
    # Preprocess #
    ##############
    train_dataset, dev_dataset, test_dataset, word_set = get_datasets(argv)
    train_samples, dev_samples, test_samples = create_samples(argv, train_dataset, dev_dataset, test_dataset)
    train_samples, dev_samples, test_samples, vocab_word = convert_samples(train_samples, dev_samples, test_samples)

    ##################
    # Create a model #
    ##################
    say('\n\nCreating a model\n')
    model = Model(docs=train_samples)
    say('\nSamples: %d  Words: %d\n' % (model.n_docs, model.n_word_types))

    ##################
    # Compute TF-IDF #
    ##################
    if dev_samples:
        say('\nDEV SET')
        model.compute(dev_samples)

    if test_samples:
        say('\nTEST SET')
        model.compute(test_samples)
