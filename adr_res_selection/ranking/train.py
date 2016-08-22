import time
import math

import numpy as np

from ..utils import say, load_dataset, load_init_emb, dump_data
from preprocessor import get_samples, theano_format
from decoder import Decoder
from eval import Evaluator


def train(argv):
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
    train_samples = theano_format(train_dataset, argv.batch)
    n_train_batches = len(train_samples)
    say('\n\nTRAIN SETTING\tBatch Size:%d  Epoch:%d  Vocab:%d  Max Words:%d  Unit:%s  Mini-Batch:%d' %
        (argv.batch, argv.epoch, token_dict.size(), max_n_words, argv.unit, n_train_batches))

    if argv.dev_data:
        dev_dataset = get_samples(threads=dev_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                  max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        dev_samples = theano_format(dev_dataset, argv.batch, True)
        say('\n\nDEV SETTING\tMini-Batch:%d' % len(dev_samples))

    if argv.test_data:
        test_dataset = get_samples(threads=test_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                   max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        test_samples = theano_format(test_dataset, argv.batch, True)
        say('\n\nTEST SETTING\tMini-Batch:%d' % len(test_samples))

    train_data = dev_data = test_data = None

    #################
    # Build a model #
    #################
    say('\n\nBuilding a model...')
    decoder = Decoder(argv=argv, emb=emb, vocab=token_dict.w2i, n_prev_sents=n_prev_sents)
    decoder.set_model()
    decoder.set_train_f()
    decoder.set_test_f()

    ###################
    # Train the model #
    ###################
    say('\nTraining start\n')

    acc_history = {}
    best_dev_acc_both = 0.
    unchanged = 0
    cand_indices = range(n_train_batches)

    for epoch in xrange(argv.epoch):
        ##############
        # Early stop #
        ##############
        unchanged += 1
        if unchanged > 15:
            say('\n\nEARLY STOP\n')
            break

        say('\n\n\nEpoch: %d' % (epoch + 1))
        say('\n  TRAIN\n  ')

        evaluator = Evaluator()

        #######################
        # Shuffle the samples #
        #######################
        train_samples = theano_format(train_dataset, argv.batch)
        np.random.shuffle(cand_indices)

        ############
        # Training #
        ############
        start = time.time()
        for index, b_index in enumerate(cand_indices):
            if index != 0 and index % 100 == 0:
                say("  {}/{}".format(index, len(cand_indices)))

            sample = train_samples[b_index]
            x = sample[0]
            binned_n_agents = sample[1]
            labels_a = sample[2]
            labels_r = sample[3]

            cost, g_norm, pred_a, pred_r = decoder.train(c=x[0], r=x[1], a=x[2], res_vec=x[3], adr_vec=x[4], n_agents=x[5])
#            cost, g_norm, pred_a, pred_r, alpha = decoder.train(c=x[0], r=x[1], a=x[2], res_vec=x[3], adr_vec=x[4], n_agents=x[5])

            if math.isnan(cost):
                say('\n\nLoss is NAN: Mini-Batch Index: %d\n' % index)
                exit()

            evaluator.update(binned_n_agents, cost, g_norm, pred_a, pred_r, labels_a, labels_r)

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        say("\n\tp_norm: {}\n".format(decoder.get_pnorm_stat()))
        evaluator.show_results()

        ############
        # Dev data #
        ############
        if argv.dev_data:
            say('\n\n  DEV\n  ')
            dev_acc_t = decoder.predict_all(dev_samples)

            if dev_acc_t > best_dev_acc_both:
                unchanged = 0
                best_dev_acc_both = dev_acc_t
                acc_history[epoch+1] = [best_dev_acc_both]

                if argv.save:
                    fn = 'Model-%s.unit-%s.attention-%d.batch-%d.reg-%f.sents-%d.words-%d' %\
                         (argv.model, argv.unit, argv.attention, argv.batch, argv.reg, n_prev_sents, max_n_words)
                    dump_data(decoder.model, fn)

        #############
        # Test data #
        #############
        if argv.test_data:
            say('\n\n\r  TEST\n  ')
            test_acc_both = decoder.predict_all(test_samples)

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
    print '\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n'
    train(argv)
