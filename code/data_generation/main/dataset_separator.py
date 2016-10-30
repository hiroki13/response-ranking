import gzip
import numpy as np


def save(fn, data):
    with gzip.open(fn + '.gz', 'wb') as gf:
        for sents in data:
            for sent in sents:
                gf.writelines("\t".join(sent) + '\n')
            gf.writelines('\n')


def main(argv):
    print '\nDATASET SEPARATION START'

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
                thread.append(line)

    print 'Threads: %d' % len(dataset)

    cand_indices = range(len(dataset))
    np.random.shuffle(cand_indices)
    dataset = [dataset[i] for i in cand_indices]

    folded = len(dataset) / 20
    train_data = dataset[:folded * 18]
    dev_data = dataset[folded * 18: folded * 19]
    test_data = dataset[folded * 19:]

    print 'Train: %d\tDev: %d\tTest: %d' % (len(train_data), len(dev_data), len(test_data))

    print 'Saving...'
    save('train-data', train_data)
    save('dev-data', dev_data)
    save('test-data', test_data)

    return dataset
