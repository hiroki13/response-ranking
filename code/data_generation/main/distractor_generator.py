import gzip
import numpy as np


def add_distractors(corpus, n_cands):
    """
    :param corpus: 1D: n_threads, 2D: n_utterances; elem=(time, speakerID, utterance)
    :return:
    """
    new_corpus = []
    pad = ['-' for i in xrange(n_cands)]
    for thread in corpus:
        new_thread = []
        indices = range(len(thread))
        agents_in_ctx = set([])
        for index, line in enumerate(thread):
            spk_id = line[1]
            adr_id = line[2]
            agents_in_ctx.add(spk_id)
            label = line[-1]

            if is_sample(spk_id, adr_id, agents_in_ctx, label):
                new_thread.append(set_line(thread, line, indices, index, n_cands))
            else:
                line = line[:4] + pad
                new_thread.append(line)
        new_corpus.append(new_thread)
    print 'Threads: %d' % len(new_corpus)
    return new_corpus


def is_sample(spk_id, adr_id, agents_in_ctx, label):
    if label == '-':
        return False
    if spk_id == adr_id:
        return False
    if adr_id not in agents_in_ctx:
        return False
    return True


def set_line(thread, line, indices, index, n_cands):
    pos_response = line[3]
    neg_responses = get_neg_responses(thread, indices, index, n_cands-1)
    responses = [pos_response] + neg_responses
    cand_indices = range(len(responses))
    np.random.shuffle(cand_indices)

    cands = [responses[index] for index in cand_indices]
    label = cand_indices.index(0)
    return line[:3] + cands + [str(label)]


def get_neg_responses(thread, indices, pos_sample_index, n_cands):
    neg_indices = list(set(indices) - set([pos_sample_index]))
    np.random.shuffle(neg_indices)
    return [thread[neg_indices[i]][3] for i in xrange(n_cands)]


def load(argv):
    dataset = []
    thread = []
    with gzip.open(argv.data, 'rb') as gf:
        for line in gf:
            line = line.rstrip().split("\t")
            if len(line) < 6:
                if thread:
                    dataset.append(thread)
                    thread = []
            else:
                pos_response = get_pos_response(line)
                line = line[:3] + [pos_response] + [line[-1]]
                thread.append(line)
    print 'Threads: %d' % len(dataset)
    return dataset


def get_pos_response(line):
    if line[-1] == '-':
        return line[3]
    cands = line[3: -1]
    label = int(line[-1])
    return cands[label]


def save(fn, data):
    with gzip.open(fn + '.gz', 'wb') as gf:
        for sents in data:
            for sent in sents:
                gf.writelines("\t".join(sent) + '\n')
            gf.writelines('\n')


def main(argv):
    print '\nADDING DISTRACTORS START'

    # corpus: 1D: n_threads, 2D: n_utterances; elem=(time, speakerID, utterance)
    corpus = load(argv)
    dataset = add_distractors(corpus, argv.n_cands)
    save(argv.fn, dataset)
