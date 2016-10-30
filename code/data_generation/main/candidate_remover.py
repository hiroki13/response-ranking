import gzip
import numpy as np


def load(argv):
    n_cands = argv.n_cands - 1
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
                line = get_removed_cand_set(line, n_cands)
                thread.append(line)

    print 'Threads: %d' % len(dataset)
    return dataset


def save(fn, data):
    with gzip.open(fn + '.gz', 'wb') as gf:
        for sents in data:
            for sent in sents:
                gf.writelines("\t".join(sent) + '\n')
            gf.writelines('\n')


def get_removed_cand_set(line, n_cands):
    if line[-1] == '-':
        return line[: 3+n_cands+1] + [line[-1]]

    cands = line[3: -1]
    label = int(line[-1])

    cand_indices = range(len(cands))
    cand_indices.remove(label)
    cand_indices = cand_indices[:n_cands]
    cand_indices.append(label)
    np.random.shuffle(cand_indices)

    cands = [cands[index] for index in cand_indices]
    label = cand_indices.index(label)
    line = line[:3] + cands + [str(label)]

    return line


def main(argv):
    print '\nCANDIDATE REMOVING START'

    print 'Loading...'
    dataset = load(argv)

    print 'Saving...'
    save('dataset.cand-%d' % argv.n_cands, dataset)
