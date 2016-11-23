import theano
import numpy as np

from ..utils.io_utils import say
from ..utils.stats import statistics
from ..ling import UNK
from sample import Sample


def convert_word_into_id(threads, vocab_word):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :param vocab_word: Vocab()
    :return: threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    """

    if threads is None:
        return None

    count = 0
    for thread in threads:
        for sent in thread:
            sent[3] = get_word_ids(tokens=sent[3], vocab_word=vocab_word)
            if sent[2] != '-':
                for i, r in enumerate(sent[4:-1]):
                    sent[4 + i] = get_word_ids(tokens=r, vocab_word=vocab_word)
                count += 1

    say('\n\tQuestions: {:>8}'.format(count))
    return threads


def get_word_ids(tokens, vocab_word):
    w_ids = []
    for token in tokens:
        if vocab_word.has_key(token):
            w_ids.append(vocab_word.get_id(token))
        else:
            w_ids.append(vocab_word.get_id(UNK))
    return w_ids


def get_samples(threads, n_prev_sents, max_n_words=1000, pad=True, test=False):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :param n_prev_sents: how many previous sentences are used
    :param max_n_words: how many words in a utterance are used
    :param pad: whether do the zero padding or not
    :param test: whether the dev/test set or not
    :return: samples: 1D: n_samples; elem=Sample()
    """

    if threads is None:
        return None

    say('\n\tTHREADS: {:>5}'.format(len(threads)))

    samples = []
    max_n_agents = n_prev_sents + 1

    for thread in threads:
        samples += get_one_thread_samples(thread, max_n_words, max_n_agents, n_prev_sents, pad, test)

    statistics(samples, max_n_agents)

    return samples


def get_one_thread_samples(thread, max_n_words, max_n_agents, n_prev_sents, pad=True, test=False):
    samples = []
    sents = []
    agents_in_ctx = set([])

    for i, sent in enumerate(thread):
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        context = get_context(i, sents, n_prev_sents, label, test)
        responses = limit_sent_length(sent[3:-1], max_n_words)

        original_sent = get_original_sent(responses, label)
        sents.append((time, spk_id, adr_id, original_sent))

        agents_in_ctx.add(spk_id)

        ################################
        # Judge if it is sample or not #
        ################################
        if is_sample(context, spk_id, adr_id, agents_in_ctx):
            sample = Sample(context=context, spk_id=spk_id, adr_id=adr_id, responses=responses, label=label,
                            n_agents_in_ctx=len(agents_in_ctx), max_n_agents=max_n_agents, max_n_words=max_n_words,
                            pad=pad, test=test)
            if test:
                samples.append(sample)
            else:
                # The num of the agents in the training samples is n_agents > 1
                # -1 means that the addressee does not appear in the limited context
                if sample.true_adr > -1:
                    samples.append(sample)

    return samples


def is_sample(context, spk_id, adr_id, agents_in_ctx):
    if context is None:
        return False
    if spk_id == adr_id:
        return False
    if adr_id not in agents_in_ctx:
        return False
    return True


def get_context(i, sents, n_prev_sents, label, test=False):
    # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens, label)
    context = None
    if label > -1:
        if len(sents) >= n_prev_sents:
            context = sents[i - n_prev_sents:i]
        elif test:
            context = sents[:i]
    return context


def get_original_sent(responses, label):
    if label > -1:
        return responses[label]
    return responses[0]


def limit_sent_length(sents, max_n_words):
    return [sent[:max_n_words] for sent in sents]


def get_max_n_words(context, response):
    return np.max([len(r) for r in response] + [len(c[-1]) for c in context])


def numpy_format(samples, batch_size, test=False):
    """
    :param samples: 1D: n_samples; elem=Sample
    :param batch_size: int
    :param test: whether dev/test or not; set boolean
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
        batch[2].append(sample.spk_agent_one_hot_vec)
        batch[3].append(sample.true_res)
        batch[4].append(batch_indexing(sample.adr_label_vec, batch_size=len(batch[4]), n_labels=n_agents - 1))

        binned_n_agents.append(sample.binned_n_agents_in_ctx)
        labels_a.append(sample.true_adr)
        labels_r.append(sample.true_res)

    if batch[0]:
        batch.append(prev_n_agents)
        batch = map(lambda s: array(s[1]) if s[0] != 2 else array(s[1], True), enumerate(batch))
        sampleset.append((batch, array(binned_n_agents), array(labels_a), array(labels_r)))

    return sampleset


def theano_shared_format(samples, batch_size):
    """
    :param samples: 1D: n_samples; elem=Sample
    :param batch_size: int
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
        batch[2].append(sample.spk_agent_one_hot_vec)
        batch[3].append(sample.true_res)
        batch[4].append(batch_indexing(sample.adr_label_vec, batch_size=len(batch[4]), n_labels=n_agents - 1))

        binned_n_agents.append(sample.binned_n_agents_in_ctx)
        labels_a.append(sample.true_adr)
        labels_r.append(sample.true_res)

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
