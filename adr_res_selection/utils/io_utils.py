import sys
import gzip
import cPickle

import numpy as np
import theano

from ..ling.vocab import Vocab, PAD, UNK


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def load_dataset(fn, vocab=set([]), check=False):
    """
    :return: samples: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None, vocab

    samples = []
    sample = []
    file_open = gzip.open if fn.endswith(".gz") else open

    with file_open(fn) as gf:
        # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
        for line in gf:
            line = line.rstrip().split("\t")
            if len(line) < 6:
                samples.append(sample)
                sample = []
            else:
                for i, sent in enumerate(line[3:-1]):
                    word_ids = []
                    for w in sent.split():
                        w = w.lower()
                        vocab.add(w)
                        word_ids.append(w)
                    line[3 + i] = word_ids

                ################################
                # Label                        #
                # -1: Not related utterances   #
                # 0-: Candidate response index #
                ################################
                line[-1] = -1 if line[-1] == '-' else int(line[-1])
                sample.append(line)
    if check:
        say('\n\n LOAD DATA EXAMPLE:\n\t%s' % str(samples[0][0]))

    return samples, vocab


def load_init_emb(init_emb, vocab_words):
    say('\nLoad Initial Word Embedding...')

    vocab = Vocab()

    ########################
    # Predefined vocab ids #
    ########################
    vocab.add_word(PAD)
    vocab.add_word(UNK)

    ################################
    # Load pre-trained  embeddings #
    ################################
    if init_emb is None:
        for w in vocab_words:
            vocab.add_word(w)
        emb = None
        say('\n\tRandom Initialized Word Embeddings')
    else:
        emb = []
        with open(init_emb) as lines:
            for line in lines:
                line = line.strip().decode('utf-8').split()
                w = line[0]
                e = line[1:]
                dim_emb = len(e)

                if dim_emb == 300 or dim_emb == 50 or dim_emb == 100 or dim_emb == 200:
                    if w in vocab_words and not vocab.has_key(w):
                        vocab.add_word(w)
                        emb.append(np.asarray(e, dtype=theano.config.floatX))

        unk = np.mean(emb, 0)
        emb.insert(0, unk)
        emb = np.asarray(emb, dtype=theano.config.floatX)

        assert emb.shape[0] == vocab.size() - 1, 'emb: %d  vocab: %d' % (emb.shape[0], vocab.size())
        say('\n\tWord Embedding Size: %d' % emb.shape[0])

    return vocab, emb


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)

