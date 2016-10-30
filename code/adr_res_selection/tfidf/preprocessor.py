from ..utils import say
from ..utils.stats import statistics
from ..ling import Vocab
from sample import Sample


def get_samples(threads, n_prev_sents, test=False):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :return: samples: 1D: n_samples; elem=Sample()
    """

    if threads is None:
        return None

    say('\n\n\tTHREADS: {:>5}'.format(len(threads)))

    samples = []
    max_n_agents = n_prev_sents + 1

    for thread in threads:
        samples += get_one_thread_samples(thread, max_n_agents, n_prev_sents, test)

    statistics(samples, max_n_agents)

    return samples


def get_one_thread_samples(thread, max_n_agents, n_prev_sents, test=False):
    samples = []
    sents = []
    agents_in_ctx = set([])

    for i, sent in enumerate(thread):
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        context = get_context(i, sents, n_prev_sents, label, test)
        responses = sent[3:-1]

        original_sent = get_original_sent(responses, label)
        sents.append((time, spk_id, adr_id, original_sent))

        agents_in_ctx.add(spk_id)

        ################################
        # Judge if it is sample or not #
        ################################
        if is_sample(context, spk_id, adr_id, agents_in_ctx):
            sample = Sample(context=context, spk_id=spk_id, adr_id=adr_id, responses=responses, label=label,
                            n_agents_in_ctx=len(agents_in_ctx), max_n_agents=max_n_agents)
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


def convert_sample_into_ids(samples, vocab_word=Vocab()):
    if samples is None:
        return None, None

    register = False
    if vocab_word.size() == 0:
        register = True

    for sample in samples:
        ctx_ids = []
        res_ids = []
        for c in sample.context:
            ctx_ids.append(convert_one_sample_into_ids(c, vocab_word, register))
        for i, r in enumerate(sample.response):
            if i == sample.true_res:
                res_ids.append(convert_one_sample_into_ids(r, vocab_word, register))
            else:
                res_ids.append(convert_one_sample_into_ids(r, vocab_word, False))
        sample.context = ctx_ids
        sample.response = res_ids

    return samples, vocab_word


def convert_one_sample_into_ids(sent, vocab_word, register):
    ids = []
    for w in sent:
        if vocab_word.has_key(w):
            w_id = vocab_word.get_id(w)
        else:
            if register:
                vocab_word.add_word(w)
                w_id = vocab_word.get_id(w)
            else:
                w_id = -1
        ids.append(w_id)
    return ids
