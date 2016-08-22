import theano
import numpy as np
from collections import defaultdict
import gzip

from sample import Sample
from ling.vocab import Vocab
from utils.io_utils import say

PAD = '<PAD>'
UNK = '<UNKNOWN>'


def load_dataset(fn, check=False, vocab=set([])):
    """
    :param fn:
    :param check:
    :param vocab:
    :return: samples: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None, vocab

    samples = []
    sample = []
    with gzip.open(fn, 'rb') as gf:
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
    say('\n\tLoad Initial Word Embedding...')

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
        say('\n\t  Random Initialized Word Embeddings')
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
        say('\n\t  Word Embedding Size: %d' % emb.shape[0])

    return vocab, emb


def set_token_id(samples, token_dict):
    def set_id(tokens):
        w_ids = []
        for token in tokens:
            if token_dict.has_key(token):
                w_ids.append(token_dict[token])
            else:
                w_ids.append(token_dict[UNK])
        return w_ids

    count = 0
    for sample in samples:
        for sent in sample:
            sent[3] = set_id(tokens=sent[3])
            if sent[2] != '-':
                for i, r in enumerate(sent[4:-1]):
                    sent[4 + i] = set_id(tokens=r)
                count += 1

    say('\n\n\tQuestions: %d' % count)
    return samples


def limit_sentence_length(sents, max_n_words, test=False):
    if test:
        return sents
    return [sent[:max_n_words] for sent in sents]


def get_samples(threads, token_dict, n_prev_sents=5, max_n_words=20, sample_size=10000000, test=False):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    """

    samples = []
    max_n_agents = n_prev_sents + 1
    threads = set_token_id(threads, token_dict)

    say('\n\tThreads: %d' % len(threads))
    for thread in threads:
        agents_in_ctx = set([])
        thread_utterances = []

        for i, sent in enumerate(thread):
            time = sent[0]
            speaker_id = sent[1]
            addressee_id = sent[2]
            responses = limit_sentence_length(sent[3:-1], max_n_words, test)
            label = sent[-1]
            context = None

            ##############################################################
            # Whether the addressee has appeared in the previous context #
            ##############################################################
            is_new_adr = False if addressee_id in agents_in_ctx else True

            ###############################################
            # Get the limited context if sent is a sample #
            ###############################################
            if label > -1:
                # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens)
                if len(thread_utterances) >= n_prev_sents:
                    context = thread_utterances[i - n_prev_sents:i]
                else:
                    if test:
                        context = thread_utterances[:i]

            ##############################################################
            # Original sentences are identical to ground-truth responses #
            ##############################################################
            if label > -1:
                thread_utterances.append((time, speaker_id, addressee_id, responses[label]))
            else:
                thread_utterances.append((time, speaker_id, addressee_id, responses[0]))

            ############################
            # Add the appeared speaker #
            ############################
            agents_in_ctx.add(speaker_id)

            ###################################
            # Add the sample into the dataset #
            ###################################
            if speaker_id != addressee_id and is_new_adr is False and context is not None:
                # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens)
                n_agents_in_lctx = len(set([c[1] for c in context] + [speaker_id]))
                n_words = get_max_n_words(context, responses)

                ###################
                # Create a sample #
                ###################
                n_words = n_words if test else max_n_words
                n_agents_in_ctx = len(agents_in_ctx)
                sample = Sample(context=context, time=time, speaker_id=speaker_id, addressee_id=addressee_id,
                                responses=responses, label=label, n_agents_in_lctx=n_agents_in_lctx,
                                binned_n_agents_in_ctx=bin_n_agents_in_ctx(n_agents_in_ctx),
                                n_agents_in_ctx=n_agents_in_ctx, max_n_agents=max_n_agents, max_n_words=n_words)

                ##################
                # Add the sample #
                ##################
                if test:
                    samples.append(sample)
                else:
                    # The num of the agents in the training samples is n_agents > 1
                    # -1 means that the addressee does not appear in the limited context
                    if sample.true_addressee > -1:
                        samples.append(sample)

            ###############################
            # Limit the number of samples #
            ###############################
            if len(samples) == sample_size:
                break

    statistics(samples)

    if test:
        samples.sort(key=lambda sample: len(sample.context[0]))  # sort with n_words
        samples.sort(key=lambda sample: len(sample.context))  # sort with n_prev_sents
    samples.sort(key=lambda sample: sample.n_agents_in_lctx)

    return samples


def get_max_n_words(context, response):
    return np.max([len(r) for r in response] + [len(c[-1]) for c in context])


def statistics(samples):
    agent_stats = defaultdict(int)
    ctx_stats = defaultdict(int)
    lctx_stats = defaultdict(int)
    adr_stats = defaultdict(int)
    na_adr_stats = defaultdict(int)
    max_n_agents = len(samples[0].context) + 1

    for sample in samples:
        agent_stats[sample.binned_n_agents_in_ctx] += 1
        ctx_stats[sample.binned_n_agents_in_ctx] += 1
        lctx_stats[sample.n_agents_in_lctx] += 1
        if sample.true_addressee > -1:
            adr_stats[sample.n_agents_in_lctx] += 1
        else:
            na_adr_stats[sample.n_agents_in_lctx] += 1

    total = 0.
    c_total = 0.

    for k, v in agent_stats.items():
        total += v
        c_total += k * v
    say('\n\t  TOTAL USED SAMPLES: %d  ADDRESSEE DETECTION CHANCE LEVEL: %f  RESPONSES: %d' %
        (total, total / c_total, len(sample.response)))

    say('\n\t  SAMPLE INFO: The Number of Agents in Limited Context:')
    for n_agents in xrange(max_n_agents):
        n_agents += 1
        if n_agents in adr_stats:
            ttl1 = adr_stats[n_agents]
        else:
            ttl1 = 0
        if n_agents in na_adr_stats:
            ttl2 = na_adr_stats[n_agents]
        else:
            ttl2 = 0
        say('\n\t\tNum %d: %d  NA-Agents: %d  TOTAL: %d' % (n_agents, ttl1, ttl2, ttl1+ttl2))

    say('\n\n\t  SAMPLE INFO: The Binned Number of Agents in Context:')
    for n_agents, ttl in sorted(ctx_stats.items(), key=lambda x: x[0]):
        say('\n\t\tNum %d: %d' % (n_agents, ttl))


def bin_n_agents_in_ctx(n):
    if n < 6:
        return 0
    elif n < 11:
        return 1
    elif n < 16:
        return 2
    elif n < 21:
        return 3
    elif n < 31:
        return 4
    elif n < 101:
        return 5
    return 6


def theano_format(samples, batch_size):
    """
    :param samples: 1D: n_samples; elem=Sample
    :param batch_size: scalar
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """

    def shared(_sample, _float=False):
        if _float:
            return theano.shared(np.asarray(_sample, dtype=theano.config.floatX), borrow=True)
        else:
            return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    sampleset = [[] for i in xrange(8)]
    batch = [[] for i in xrange(8)]
    n_agents_in_limited_ctx = []
    n_agents_in_ctx = []
    prev_n_agents = samples[0].n_agents_in_lctx

    for sample in samples:
        n_agents = sample.n_agents_in_lctx

        if len(batch[0]) == batch_size:
            sampleset[0].append(batch[0])
            sampleset[1].append(batch[1])
            sampleset[2].append(batch[2])
            sampleset[3].append(batch[3])
            sampleset[4].append(batch[4])
            sampleset[5].append(batch[5])
            sampleset[6].append(batch[6])
            sampleset[7].append(prev_n_agents)
            n_agents_in_limited_ctx.append(prev_n_agents)
            n_agents_in_ctx.append(batch[7])
            batch = [[] for i in xrange(8)]

        if n_agents != prev_n_agents:
            batch = [[] for i in xrange(8)]
            prev_n_agents = n_agents

        batch[0].append(sample.context)
        batch[1].append(sample.response)
        batch[2].append(sample.speaking_agent_one_hot_vector)
        batch[3].append(sample.true_response)
        batch[4].append(sample.false_response)
        batch[5].append(sample.true_addressee)
        batch[6].append(sample.false_addressee)
        batch[7].append(sample.binned_n_agents_in_ctx)

    n_batches = len(sampleset[-1])
    shared_sampleset = map(lambda s: shared(s[1]) if s[0] != 2 else shared(s[1], True), enumerate(sampleset))
    return shared_sampleset, n_agents_in_limited_ctx, n_batches, n_agents_in_ctx


def theano_format_test(samples):
    """
    :param samples: 1D: n_samples; elem=Sample
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """

    def array(_sample, _float=False):
        if _float:
            return np.asarray(_sample, dtype=theano.config.floatX)
        else:
            return np.asarray(_sample, dtype='int32')

    sampleset = [[] for i in xrange(8)]
    batch = [[] for i in xrange(8)]
    n_agents_in_limited_ctx = []
    n_agents_in_ctx = []
    prev_n_agents = samples[0].n_agents_in_lctx
    prev_n_sents = len(samples[0].context)
    prev_n_words = len(samples[0].context[0])

    for i in xrange(len(samples) + 1):
        if i < len(samples):
            sample = samples[i]
            n_agents = sample.n_agents_in_lctx
            n_sents = len(sample.context)
            n_words = len(sample.context[0])

        if i == len(samples) or n_agents != prev_n_agents or n_sents != prev_n_sents or n_words != prev_n_words:
            batch = map(lambda s: array(s[1]) if s[0] != 2 else array(s[1], True), enumerate(batch))
            sampleset[0].append(batch[0])
            sampleset[1].append(batch[1])
            sampleset[2].append(batch[2])
            sampleset[3].append(batch[3])
            sampleset[4].append(batch[4])
            sampleset[5].append(batch[5])
            sampleset[6].append(batch[6])
            sampleset[7].append(prev_n_agents)
            n_agents_in_limited_ctx.append(prev_n_agents)
            n_agents_in_ctx.append(batch[7])

            batch = [[] for i in xrange(8)]
            prev_n_agents = n_agents
            prev_n_sents = n_sents
            prev_n_words = n_words

        batch[0].append(sample.context)
        batch[1].append(sample.response)
        batch[2].append(sample.speaking_agent_one_hot_vector)
        batch[3].append(sample.true_response)
        batch[4].append(sample.false_response)
        batch[5].append(sample.true_addressee)
        batch[6].append(sample.false_addressee)
        batch[7].append(sample.binned_n_agents_in_ctx)

    n_batches = len(sampleset[-1])
    return sampleset, n_agents_in_limited_ctx, n_batches, n_agents_in_ctx
