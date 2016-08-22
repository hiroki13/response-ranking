import numpy as np


class Sample(object):

    def __init__(self, context, time, speaker_id, addressee_id, responses, label,
                 n_agents_in_lctx, binned_n_agents_in_ctx, n_agents_in_ctx, max_n_agents, max_n_words):

        self.orig_context = context
        self.time = time
        self.speaker_id = speaker_id
        self.addressee_id = addressee_id
        self.responses = responses
        self.label = label

        agent_index_dict = indexing(speaker_id, context)

        # 1D: n_prev_sents, 2D: max_n_words
        self.context = padding_context(context, max_n_words)

        # 1D: n_cands, 2D: max_n_words
        self.response = padding_response(responses, max_n_words)

        # 1D: n_prev_sents, 2D: max_n_agents; one-hot vector
        self.speaking_agent_one_hot_vector = get_speaking_agent_one_hot_vector(context, agent_index_dict, max_n_agents)

        self.true_response = label
        self.false_response = get_false_response_label(responses, label)

        adr_labels = get_addressee_labels(addressee_id, agent_index_dict)
        self.true_addressee = adr_labels[0]
        self.false_addressee = adr_labels[1]

        self.n_agents_in_lctx = n_agents_in_lctx
        self.binned_n_agents_in_ctx = binned_n_agents_in_ctx
        self.n_agents_in_ctx = n_agents_in_ctx


def get_addressee_labels(addressee_id, agent_index_dict):
    """
    :param addressee_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if addressee_id in agent_index_dict and n_agents_lctx > 1:
        true_addressee = agent_index_dict[addressee_id] - 1

        # responding agent id == addressee agent id
        cand_indices = range(n_agents_lctx - 1)
        cand_indices.remove(true_addressee)

        # Negative addressee candidates do not exist
        if len(cand_indices) == 0:
            false_addressee = 0
        else:
            np.random.shuffle(cand_indices)
            false_addressee = cand_indices[0]
    else:
        true_addressee = -1
        false_addressee = -1

    return true_addressee, false_addressee


def get_addressee_label_vector(adr_id, agent_index_dict):
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

    return y


def get_false_response_label(response, label):
    """
    :param response: [response1, response2, ... ]
    :param label: true response label; int
    :return: int
    """
    n_responses = len(response)
    cand_indices = range(n_responses)
    cand_indices.remove(label)
    np.random.shuffle(cand_indices)
    return cand_indices[0]


def get_speaking_agent_one_hot_vector(context, agent_index_dict, max_n_agents):
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

