import theano
import numpy as np
from collections import defaultdict

from ..utils import say
from ..ling import UNK
from sample import Sample


def convert_word_into_id(threads, vocab):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :return: threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    """

    if threads is None:
        return None

    def get_word_ids(tokens):
        w_ids = []
        for token in tokens:
            if vocab.has_key(token):
                w_ids.append(vocab.get_id(token))
            else:
                w_ids.append(vocab.get_id(UNK))
        return w_ids

    count = 0
    for thread in threads:
        for sent in thread:
            sent[3] = get_word_ids(tokens=sent[3])
            if sent[2] != '-':
                for i, r in enumerate(sent[4:-1]):
                    sent[4 + i] = get_word_ids(tokens=r)
                count += 1

    say('\n\tQuestions: %d' % count)
    return threads


def get_samples(threads, n_prev_sents, max_n_words=20, sample_size=10000000, test=False):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :return: samples: 1D: n_samples; elem=Sample()
    """

    if threads is None:
        return None

    samples = []
    max_n_agents = n_prev_sents + 1

    say('\n\n\tThreads: %d' % len(threads))
    for thread in threads:
        agents_in_ctx = set([])
        thread_utterances = []

        for i, sent in enumerate(thread):
            time = sent[0]
            speaker_id = sent[1]
            addressee_id = sent[2]
            responses = limit_sentence_length(sent[3:-1], max_n_words)
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

                ###################
                # Create a sample #
                ###################
                n_agents_in_ctx = len(agents_in_ctx)
                sample = Sample(context=context, time=time, speaker_id=speaker_id, addressee_id=addressee_id,
                                responses=responses, label=label, n_agents_in_lctx=n_agents_in_lctx,
                                n_agents_in_ctx=n_agents_in_ctx, max_n_agents=max_n_agents, max_n_words=max_n_words)

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

    statistics(samples, max_n_agents)

    return samples


def limit_sentence_length(sents, max_n_words):
    return [sent[:max_n_words] for sent in sents]


def get_max_n_words(context, response):
    return np.max([len(r) for r in response] + [len(c[-1]) for c in context])


def statistics(samples, max_n_agents):
    agent_stats = defaultdict(int)
    ctx_stats = defaultdict(int)
    lctx_stats = defaultdict(int)
    adr_stats = defaultdict(int)
    na_adr_stats = defaultdict(int)

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


def theano_format(samples, batch_size, n_cands, test=False):
    """
    :param samples: 1D: n_samples; elem=Sample
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """

    def array(_sample, _float=False):
        if _float:
            return np.asarray(_sample, dtype=theano.config.floatX)
        else:
            return np.asarray(_sample, dtype='int32')

    if samples is None:
        return None

    ########
    # Sort #
    ########
    if test:
        samples.sort(key=lambda sample: len(sample.context[0]))  # sort with n_words
        samples.sort(key=lambda sample: len(sample.context))  # sort with n_prev_sents
    samples.sort(key=lambda sample: sample.n_agents_in_lctx)

    if test is False:
        samples = shuffle_batch(samples)

    ##############
    # Initialize #
    ##############
    sampleset = []

    batch = [[] for i in xrange(5)]
    binned_n_agents = []
    labels_a = []
    labels_r = []

    prev_n_agents = samples[0].n_agents_in_lctx
    prev_n_prev_sents = len(samples[0].context)

    for sample in samples:
        n_agents = sample.n_agents_in_lctx
        n_prev_sents = len(sample.context)

        if len(batch[0]) == batch_size or n_agents != prev_n_agents or n_prev_sents != prev_n_prev_sents:
            assert len(batch[0]) == len(binned_n_agents) == len(labels_a) == len(labels_r)

            batch.append(prev_n_agents)

            # batch: (c, r, a, res_vec, adr_vec, n_agents)
            batch = map(lambda s: array(s[1]) if s[0] != 2 else array(s[1], True), enumerate(batch))

            sampleset.append((batch, array(binned_n_agents), array(labels_a), array(labels_r)))
            prev_n_agents = n_agents
            prev_n_prev_sents = n_prev_sents

            ##############
            # Initialize #
            ##############
            batch = [[] for i in xrange(5)]
            binned_n_agents = []
            labels_a = []
            labels_r = []

        ##################
        # Create a batch #
        ##################
        batch[0].append(sample.context)
        batch[1].append(sample.response)
        batch[2].append(sample.speaking_agent_one_hot_vector)
        batch[3].append(batch_indexing(sample.res_label_vec, batch_size=len(batch[3]), n_labels=n_cands))
        batch[4].append(batch_indexing(sample.adr_label_vec, batch_size=len(batch[4]), n_labels=n_agents - 1))

        binned_n_agents.append(sample.binned_n_agents_in_ctx)
        labels_a.append(sample.true_addressee)
        labels_r.append(sample.true_response)

    if batch[0]:
        batch.append(prev_n_agents)
        batch = map(lambda s: array(s[1]) if s[0] != 2 else array(s[1], True), enumerate(batch))
        sampleset.append((batch, array(binned_n_agents), array(labels_a), array(labels_r)))

    return sampleset


def theano_format_shared(samples, batch_size, n_cands):
    """
    :param samples: 1D: n_samples; elem=Sample
    :return shared_sampleset: 1D: 8, 2D: n_baches, 3D: batch
    """

    def shared(_sample, _float=False):
        if _float:
            return theano.shared(np.asarray(_sample, dtype=theano.config.floatX), borrow=True)
        else:
            return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    samples.sort(key=lambda sample: sample.n_agents_in_lctx)
    samples = shuffle_batch(samples)

    ##############
    # Initialize #
    ##############
    sampleset = [[] for i in xrange(6)]
    evalset = []
    batch = [[] for i in xrange(5)]
    binned_n_agents = []
    labels_a = []
    labels_r = []

    prev_n_agents = samples[0].n_agents_in_lctx

    for sample in samples:
        n_agents = sample.n_agents_in_lctx

        if len(batch[0]) == batch_size:
            sampleset[0].append(batch[0])
            sampleset[1].append(batch[1])
            sampleset[2].append(batch[2])
            sampleset[3].append(batch[3])
            sampleset[4].append(batch[4])
            sampleset[5].append(prev_n_agents)

            evalset.append((binned_n_agents, labels_a, labels_r))

            batch = [[] for i in xrange(5)]
            binned_n_agents = []
            labels_a = []
            labels_r = []

        if n_agents != prev_n_agents:
            batch = [[] for i in xrange(5)]
            prev_n_agents = n_agents

        ##################
        # Create a batch #
        ##################
        batch[0].append(sample.context)
        batch[1].append(sample.response)
        batch[2].append(sample.speaking_agent_one_hot_vector)
        batch[3].append(batch_indexing(sample.res_label_vec, batch_size=len(batch[3]), n_labels=n_cands))
        batch[4].append(batch_indexing(sample.adr_label_vec, batch_size=len(batch[4]), n_labels=n_agents - 1))

        binned_n_agents.append(sample.binned_n_agents_in_ctx)
        labels_a.append(sample.true_addressee)
        labels_r.append(sample.true_response)

    n_batches = len(sampleset[-1])
    shared_sampleset = map(lambda s: shared(s[1]) if s[0] != 2 else shared(s[1], True), enumerate(sampleset))

    return shared_sampleset, n_batches, evalset


def batch_indexing(vec, batch_size, n_labels):
    indexed_vec = []
    for v in vec:
        if v > -1:
            v += batch_size * n_labels
        else:
            v = -1
        indexed_vec.append(v)
    return indexed_vec


def shuffle_batch(samples):
    """
    :param samples: 1D: n_samples; elem=Sample
    :return: 1D: n_samples; elem=Sample
    """

    shuffled_samples = []
    batch = []
    prev_n_agents = samples[0].n_agents_in_lctx
    prev_n_prev_sents = len(samples[0].context)

    for sample in samples:
        n_agents = sample.n_agents_in_lctx
        n_prev_sents = len(sample.context)

        if n_agents != prev_n_agents or n_prev_sents != prev_n_prev_sents:
            np.random.shuffle(batch)
            shuffled_samples += batch

            batch = []
            prev_n_agents = n_agents
            prev_n_prev_sents = n_prev_sents

        batch.append(sample)

    if batch:
        np.random.shuffle(batch)
        shuffled_samples += batch

    return shuffled_samples
