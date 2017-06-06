import os
import sys
import shutil
import gzip
import cPickle

import numpy as np
import theano

from ..ling.vocab import Vocab, PAD, UNK


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def load_dataset(fn, vocab=set([]), data_size=1000000, test=False):
    """
    :param fn: file name
    :param vocab: vocab set
    :param data_size: how many threads are used
    :return: threads: 1D: n_threads, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None, vocab

    threads = []
    thread = []
    file_open = gzip.open if fn.endswith(".gz") else open

    with file_open(fn) as gf:
        # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
        for line in gf:
            line = line.rstrip().split("\t")

            if len(line) < 6:
                threads.append(thread)
                thread = []

                if len(threads) >= data_size:
                    break
            else:
                for i, sent in enumerate(line[3:-1]):
                    words = []
                    for w in sent.split():
                        w = w.lower()
                        if test is False:
                            vocab.add(w)
                        words.append(w)
                    line[3 + i] = words

                ##################
                # Label          #
                # -1: Not sample #
                # 0-: Sample     #
                ##################
                line[-1] = -1 if line[-1] == '-' else int(line[-1])
                thread.append(line)

    return threads, vocab


def load_init_emb(init_emb, word_set=None):
    """
    :param init_emb: Column 0 = word, Column 1- = value; e.g., [the 0.418 0.24968 -0.41242 ...]
    :param word_set: the set of words that appear in train/dev/test set
    :return: vocab_word: Vocab()
    :return: emb: np.array
    """

    say('\nLoad initial word embedding...')

    ##################
    # Set vocab word #
    ##################
    vocab_word = Vocab()
    vocab_word.add_word(PAD)
    vocab_word.add_word(UNK)

    ################################
    # Load pre-trained  embeddings #
    ################################
    if init_emb is None:
        for w in word_set:
            vocab_word.add_word(w)
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
                    if not vocab_word.has_key(w):
                        vocab_word.add_word(w)
                        emb.append(np.asarray(e, dtype=theano.config.floatX))

        unk = np.mean(emb, 0)
        emb.insert(0, unk)
        emb = np.asarray(emb, dtype=theano.config.floatX)

        assert emb.shape[0] == vocab_word.size() - 1, 'emb: %d  vocab: %d' % (emb.shape[0], vocab_word.size())
        say('\n\tWord Embedding Size: %d' % emb.shape[0])

    return vocab_word, emb


def load_multi_ling_init_emb(init_emb, target_lang):
    """
    :param init_emb: Column 0 = word, Column 1- = value; e.g., [the 0.418 0.24968 -0.41242 ...]
    :return: vocab_word: Vocab()
    :return: emb: np.array
    """

    say('\nLoad initial word embedding...')

    ##################
    # Set vocab word #
    ##################
    vocab_word = Vocab()
    vocab_word.add_word(PAD)
    vocab_word.add_word(UNK)

    emb = []
    with open(init_emb) as lines:
        for line in lines:
            line = line.strip().decode('utf-8').split()
            lang = line[0][:2]
            w = line[0][3:]
            if lang != target_lang:
                continue
            e = line[1:]
            if not vocab_word.has_key(w):
                vocab_word.add_word(w)
                emb.append(np.asarray(e, dtype=theano.config.floatX))

    unk = np.mean(emb, 0)
    emb.insert(0, unk)
    emb = np.asarray(emb, dtype=theano.config.floatX)

    assert emb.shape[0] == vocab_word.size() - 1, 'emb: %d  vocab: %d' % (emb.shape[0], vocab_word.size())
    say('\n\tWord Embedding Size: %d' % emb.shape[0])

    return vocab_word, emb


def output_samples(fn, samples, vocab_word=None):
    if samples is None:
        return

    with gzip.open(fn + '.gz', 'wb') as gf:
        for sample in samples:
            agent_index_dict = [None for i in xrange(len(sample.agent_index_dict))]
            for k, v in sample.agent_index_dict.items():
                agent_index_dict[v] = k

            for a, c in zip(sample.spk_agents, sample.context):
                text = '%s\t-\t' % agent_index_dict[a]
                text += ' '.join([get_word(w, vocab_word) for w in c])
                print >> gf, text

            text = '%s\t%s\t' % (sample.spk_id, sample.adr_id)
            for r in sample.response:
                text += '%s\t' % ' '.join([get_word(w, vocab_word) for w in r])
            text += '%d' % sample.true_res
            print >> gf, text
            print >> gf


def get_word(w, vocab_word=None):
    if vocab_word:
        return vocab_word.get_word(w)
    return w


def dump_data(data, fn):
    fn = check_identifier(fn)
    with gzip.open(fn, 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    fn = check_identifier(fn)
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


def create_path(output_path):
    path = ''
    dir_names = output_path.split('/')
    for dir_name in dir_names:
        path += dir_name
        if not os.path.exists(path):
            os.mkdir(path)
        path += '/'


def move_data(src, dst):
    if not os.path.exists(dst):
        path = ''
        for dn in dst.split('/'):
            path += dn + '/'
            if os.path.exists(path):
                continue
            os.mkdir(path)
    if os.path.exists(os.path.join(dst, src)):
        os.remove(os.path.join(dst, src))
    shutil.move(src, dst)


def check_identifier(fn):
    if not fn.endswith(".pkl.gz"):
        fn += ".gz" if fn.endswith(".pkl") else ".pkl.gz"
    return fn

