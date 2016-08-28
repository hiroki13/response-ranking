import os

from ..utils import say, load_dataset, load_init_emb
from model_api import ModelAPI
from preprocessor import convert_word_into_id, get_samples, theano_format, theano_format_shared


def get_datasets(argv):
    say('\nSET UP DATASET\n')

    ###########################
    # Task setting parameters #
    ###########################
    n_cands = argv.n_cands
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    say('\nTASK  SETTING')
    say('\n\tResponse Candidates:%d  Contexts:%d  Max Word Num:%d' % (n_cands, n_prev_sents, max_n_words))

    #################
    # Load datasets #
    #################
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\n\nLoad dataset...')
    train_dataset, word_set = load_dataset(fn=argv.train_data, check=argv.check)
    dev_dataset, word_set = load_dataset(fn=argv.dev_data, vocab=word_set, check=argv.check)
    test_dataset, word_set = load_dataset(fn=argv.test_data, vocab=word_set, check=argv.check)

    return train_dataset, dev_dataset, test_dataset, word_set


def create_samples(argv, train_dataset, dev_dataset, test_dataset, vocab_words):
    n_cands = argv.n_cands
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    batch_size = argv.batch
    sample_size = argv.sample_size

    ##########################
    # Convert words into ids #
    ##########################
    say('\n\nConverting words into ids...')
    # dataset: 1D: n_samples; 2D: Sample
    train_samples = convert_word_into_id(train_dataset, vocab_words)
    dev_samples = convert_word_into_id(dev_dataset, vocab_words)
    test_samples = convert_word_into_id(test_dataset, vocab_words)

    ##################
    # Create samples #
    ##################
    say('\n\nCreating samples...')
    train_samples = get_samples(threads=train_samples, n_prev_sents=n_prev_sents,
                                max_n_words=max_n_words, sample_size=sample_size)
    dev_samples = get_samples(threads=dev_samples, n_prev_sents=n_prev_sents,
                              max_n_words=max_n_words, sample_size=sample_size, test=True)
    test_samples = get_samples(threads=test_samples, n_prev_sents=n_prev_sents,
                               max_n_words=max_n_words, sample_size=sample_size, test=True)

    ###################################
    # Create theano-formatted samples #
    ###################################
    train_samples, n_train_batches, evalset = theano_format_shared(train_samples, batch_size, n_cands=n_cands)
    dev_samples = theano_format(dev_samples, batch_size, n_cands=n_cands, test=True)
    test_samples = theano_format(test_samples, batch_size, n_cands=n_cands, test=True)

    say('\n\nTRAIN SETTING\tBatch Size:%d  Epoch:%d  Vocab:%d  Max Words:%d' %
        (batch_size, argv.epoch, vocab_words.size(), max_n_words))
    say('\n\nTrain samples\tMini-Batch:%d' % n_train_batches)
    if dev_samples:
        say('\nDev samples\tMini-Batch:%d' % len(dev_samples))
    if test_samples:
        say('\nTest samples\tMini-Batch:%d' % len(test_samples))
    return train_samples, dev_samples, test_samples, n_train_batches, evalset


def train(argv, model_api, n_train_batches, evalset, dev_samples, test_samples):
    say('\n\nTRAINING START\n')

    acc_history = {}
    best_dev_acc_both = 0.
    unchanged = 0

    batch_indices = range(n_train_batches)

    for epoch in xrange(argv.epoch):
        say('\n\n')
        os.system('free -m')

        ##############
        # Early stop #
        ##############
        unchanged += 1
        if unchanged > 15:
            say('\n\nEARLY STOP\n')
            break

        ############
        # Training #
        ############
        say('\n\n\nEpoch: %d' % (epoch + 1))
        say('\n  TRAIN\n  ')

        model_api.train_all(batch_indices, evalset)

        ##############
        # Validating #
        ##############
        if dev_samples:
            say('\n\n  DEV\n  ')
            dev_acc_both = model_api.predict_all(dev_samples)

            if dev_acc_both > best_dev_acc_both:
                unchanged = 0
                best_dev_acc_both = dev_acc_both
                acc_history[epoch+1] = [best_dev_acc_both]

                if argv.save:
                    model_api.save_model()

        if test_samples:
            say('\n\n\r  TEST\n  ')
            test_acc_both = model_api.predict_all(test_samples)

            if unchanged == 0:
                if epoch+1 in acc_history:
                    acc_history[epoch+1].append(test_acc_both)
                else:
                    acc_history[epoch+1] = [test_acc_both]

        #####################
        # Show best results #
        #####################
        say('\n\n\tBEST ACCURACY HISTORY')
        for k, v in sorted(acc_history.items()):
            if len(v) == 2:
                say('\n\tEPOCH-{:d}  \tBEST DEV ACC:{:.2%}\tBEST TEST ACC:{:.2%}'.format(k, v[0], v[1]))
            else:
                say('\n\tEPOCH-{:d}  \tBEST DEV ACC:{:.2%}'.format(k, v[0]))


def main(argv):
    say('\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n')

    ###############
    # Set samples #
    ###############
    train_dataset, dev_dataset, test_dataset, word_set = get_datasets(argv)
    vocab_words, init_emb = load_init_emb(argv.init_emb, word_set)
    train_samples, dev_samples, test_samples, n_train_batches, evalset =\
        create_samples(argv, train_dataset, dev_dataset, test_dataset, vocab_words)
    del train_dataset
    del dev_dataset
    del test_dataset

    ###############
    # Set a model #
    ###############
    model_api = ModelAPI(argv, init_emb, vocab_words, argv.n_prev_sents)
    model_api.set_model()
    model_api.set_train_f(train_samples)
    model_api.set_test_f()

    train(argv, model_api, n_train_batches, evalset, dev_samples, test_samples)
