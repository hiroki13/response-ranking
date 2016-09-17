import numpy as np


class Sample(object):

    def __init__(self, context, spk_id, adr_id, responses, label, n_agents_in_ctx, max_n_agents):

        # str
        self.spk_id = spk_id
        self.adr_id = adr_id

        # 1D: n_prev_sents, 2D: max_n_words
        self.context = [c[-1] for c in context]
        # 1D: n_cands, 2D: max_n_words
        self.response = responses

        self.agent_index_dict = indexing(spk_id, context)
        # 1D: n_prev_sents, 2D: max_n_agents; one-hot vector
        self.spk_agent_one_hot_vec = get_spk_agent_one_hot_vec(context, self.agent_index_dict, max_n_agents)
        self.spk_agents = [s.index(1) for s in self.spk_agent_one_hot_vec]

        self.true_res = label
        self.true_adr = get_adr_label(adr_id, self.agent_index_dict)

        self.n_agents_in_lctx = len(set([c[1] for c in context] + [spk_id]))
        self.binned_n_agents_in_ctx = bin_n_agents_in_ctx(n_agents_in_ctx)
        self.n_agents_in_ctx = n_agents_in_ctx


def get_adr_label(addressee_id, agent_index_dict):
    """
    :param addressee_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if addressee_id in agent_index_dict and n_agents_lctx > 1:
        true_addressee = agent_index_dict[addressee_id] - 1
    else:
        true_addressee = -1

    return true_addressee


def get_adr_label_vec(adr_id, agent_index_dict, max_n_agents):
    """
    :param adr_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    y = []
    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if adr_id in agent_index_dict and n_agents_lctx > 1:
        # True addressee index
        y.append(agent_index_dict[adr_id]-1)

        # False addressee index
        for i in xrange(len(agent_index_dict)-1):
            if i not in y:
                y.append(i)

    pad = [-1 for i in xrange(max_n_agents-1-len(y))]
    y = y + pad
    return y


def get_false_res_label(response, label):
    """
    :param response: [response1, response2, ... ]
    :param label: true response label; int
    :return: int
    """
    n_responses = len(response)
    cand_indices = range(n_responses)
    cand_indices.remove(label)
    np.random.shuffle(cand_indices)
    return cand_indices


def get_spk_agent_one_hot_vec(context, agent_index_dict, max_n_agents):
    """
    :param context: 1D: n_prev_sents, 2D: n_words
    :param agent_index_dict: {agent id: agent index}
    :param max_n_agents: the max num of agents that appear in the context (=n_prev_sents+1); int
    :return: 1D: n_prev_sents, 2D: max_n_agents
    """
    speaking_agent_one_hot_vector = []
    for c in context:
        vec = [0 for i in xrange(max_n_agents)]
        speaker_id = c[1]
        vec[agent_index_dict[speaker_id]] = 1
        speaking_agent_one_hot_vector.append(vec)
    return speaking_agent_one_hot_vector


def indexing(responding_agent_id, context):
    agent_ids = {responding_agent_id: 0}
    for c in reversed(context):
        agent_id = c[1]
        if not agent_ids.has_key(agent_id):
            agent_ids[agent_id] = len(agent_ids)
    return agent_ids


def padding_response(responses, max_n_words):
    pads = []
    for sent in responses:
        diff = max_n_words - len(sent)
        pad = [0 for i in xrange(diff)]
        pad.extend(sent)
        pads.append(pad)
    return pads


def padding_context(context, max_n_words):
    def padding_sent(_sent):
        diff = max_n_words - len(_sent)
        return [0 for i in xrange(diff)] + _sent
    return [padding_sent(sent[-1]) for sent in context]


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

