import os

import theano.tensor as T
from ..utils import say, load_dataset, load_init_emb
from model import StaticModel, DynamicModel
from preprocessor import get_samples, theano_format, theano_format_shared


def train(argv):
    say('\n\n')
    os.system("ps -p %d u" % os.getpid())

    say('\nSET UP TRAINING SETTINGS\n')

    ###########################
    # Task setting parameters #
    ###########################
    n_cands = argv.n_cands
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    say('\n\nTASK  SETTING')
    say('\n  Response Candidates:%d  Contexts:%d  Max Word Num:%d' % (n_cands, n_prev_sents, max_n_words))

    #################
    # Load datasets #
    #################
    # data: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\n\nLoad data sets...')
    train_data, vocab_words = load_dataset(argv.train_data, argv.check)
    dev_data, vocab_words = load_dataset(argv.dev_data, argv.check, vocab_words)
    test_data, vocab_words = load_dataset(argv.test_data, argv.check, vocab_words)

    ################################
    # Load initial word embeddings #
    ################################
    say('\n\nSet initial word embeddings...')
    token_dict, emb = load_init_emb(argv.init_emb, vocab_words)

    ################################
    # Set up experimental settings #
    ################################
    # dataset: 1D: n_samples; 2D: Sample
    say('\n\nSetting datasets...')

    train_dataset = get_samples(threads=train_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                max_n_words=max_n_words, test=False, sample_size=argv.sample_size)
    train_samples, n_train_batches, evalset = theano_format_shared(train_dataset, argv.batch, n_cands=n_cands)
    del train_dataset
    say('\n\nTRAIN SETTING\tBatch Size:%d  Epoch:%d  Vocab:%d  Max Words:%d  Unit:%s  Mini-Batch:%d' %
        (argv.batch, argv.epoch, token_dict.size(), max_n_words, argv.unit, n_train_batches))

    if argv.dev_data:
        dev_dataset = get_samples(threads=dev_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                  max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        dev_samples = theano_format(dev_dataset, argv.batch, n_cands=n_cands, test=True)
        del dev_dataset
        say('\n\nDEV SETTING\tMini-Batch:%d' % len(dev_samples))

    if argv.test_data:
        test_dataset = get_samples(threads=test_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                   max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        test_samples = theano_format(test_dataset, argv.batch, n_cands=n_cands, test=True)
        del test_dataset
        say('\n\nTEST SETTING\tMini-Batch:%d' % len(test_samples))

    ###############
    # Set a model #
    ###############
    say('\n\tMODEL: %s  Unit: %s  Opt: %s  Activation: %s\n' % (argv.model, argv.unit, argv.opt, argv.activation))

    if argv.model == 'static':
        model = StaticModel(argv=argv, max_n_agents=n_prev_sents+1, n_vocab=token_dict.size(), init_emb=emb)
    else:
        model = DynamicModel(argv=argv, max_n_agents=n_prev_sents+1, n_vocab=token_dict.size(), init_emb=emb)

    c = T.itensor3('c')
    r = T.itensor3('r')
    a = T.ftensor3('a')
    y_r = T.imatrix('y_r')
    y_a = T.imatrix('y_a')
    n_agents = T.iscalar('n_agents')

    model.compile(c, r, a, y_r, y_a, n_agents)

    model.train(train_samples,
                n_train_batches,
                evalset,
                dev_samples if argv.dev_data else None,
                test_samples if argv.test_data else None
                )


def main(argv):
    print '\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n'
    train(argv)
