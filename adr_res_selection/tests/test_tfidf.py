import numpy as np
import theano

from ..ranking.tfidf import get_datasets, create_samples, convert_samples, Model
from ..utils import say

theano.config.floatX = 'float32'
np.random.seed(0)


def main(argv):
    train_dataset, dev_dataset, test_dataset, word_set = get_datasets(argv)
    train_samples, dev_samples, test_samples = create_samples(argv, train_dataset, dev_dataset, test_dataset)
    train_samples = [train_samples[0]]

    ctx = []
    for c in train_samples[0].context:
        ctx.append(c[:3])
    res = []
    for r in train_samples[0].response:
        res.append(r[:3])

    train_samples[0].context = ctx
    train_samples[0].response = res

    train_samples, dev_samples, test_samples, vocab_word = convert_samples(train_samples, dev_samples, test_samples)

    ##################
    # Create a model #
    ##################
    say('\n\nCreating a model\n')
    model = Model(docs=train_samples)
    say('\nSamples: %d  Words: %d\n' % (model.n_docs, model.n_word_types))

    ##################
    # Compute TF-IDF #
    ##################
    if dev_samples:
#        model.compute(dev_samples)
        model.compute(train_samples)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger')

    ################
    # Main options #
    ################
    parser.add_argument('--train_data',  help='path to data')
    parser.add_argument('--dev_data',  help='path to data')
    parser.add_argument('--test_data',  help='path to data')

    #######################
    # Training parameters #
    #######################
    parser.add_argument('--n_prev_sents', type=int, default=5, help='prev sents')
    parser.add_argument('--max_n_words', type=int, default=20, help='maximum number of words for context/response')
    parser.add_argument('--n_cands', type=int, default=2, help='number of candidate responses')
    parser.add_argument('--sample_size', type=int, default=10000000, help='sample size')
    parser.add_argument('--check',  default=False, help='check the code')

    #################
    # Starting mode #
    #################
    argv = parser.parse_args()
    print
    print argv
    print

    main(argv)
