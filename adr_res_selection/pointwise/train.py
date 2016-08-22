import sys
import time

from utils import io_utils
from preprocessor import get_samples, load_dataset, load_init_emb, theano_format, theano_format_test
from model_builder import set_model
from decoder import Decoder
from utils.io_utils import say
from evaluator import Evaluator

import numpy as np


def train(argv):
    print 'SET UP TRAINING SETTINGS'

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
    if argv.save:
        io_utils.dump_data(data=token_dict, fn='vocab')

    ################################
    # Set up experimental settings #
    ################################
    # dataset: 1D: n_samples; 2D: Sample
    say('\n\nSetting datasets...')

    train_dataset = get_samples(threads=train_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                max_n_words=max_n_words, test=False, sample_size=argv.sample_size)
    tr_samples, tr_n_agents, tr_n_batches, tr_n_thread_agents = theano_format(train_dataset, argv.batch)
    say('\n\nTRAIN SETTING\tBatch Size:%d  Epoch:%d  Vocab:%d  Max Words:%d  Unit:%s  Mini-Batch:%d' %
        (argv.batch, argv.epoch, token_dict.size(), max_n_words, argv.unit, tr_n_batches))

    if argv.dev_data:
        dev_dataset = get_samples(threads=dev_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                  max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        dev_samples, dev_n_agents, dev_n_batches, dev_n_thread_agents = theano_format_test(dev_dataset)
        say('\n\nDEV SETTING\tMini-Batch:%d' % dev_n_batches)

    if argv.test_data:
        test_dataset = get_samples(threads=test_data, token_dict=token_dict.w2i, n_prev_sents=n_prev_sents,
                                   max_n_words=max_n_words, test=True, sample_size=argv.sample_size)
        test_samples, test_n_agents, test_n_batches, test_n_thread_agents = theano_format_test(test_dataset)
        say('\n\nTEST SETTING\tMini-Batch:%d' % test_n_batches)

    #################
    # Build a model #
    #################
    say('\n\nBuilding a model...')
    model = set_model(argv=argv, emb=emb, vocab=token_dict.w2i, n_prev_sents=n_prev_sents)

    #################
    # Set a decoder #
    #################
    decoder = Decoder(argv=argv, model=model)
    decoder.set_train_f(tr_samples)
    decoder.set_test_f()

    ###################
    # Train the model #
    ###################
    say('\nTraining start\n')

    evaluator = Evaluator()
    best_acc = 0.
    best_epoch = 0
    cand_indices = range(tr_n_batches)

    for epoch in xrange(argv.epoch):
        say('\n\n\nEpoch: %d' % (epoch + 1))
        say('\n  TRAIN\n  ')

        best = False
        start = time.time()
        np.random.shuffle(cand_indices)

        for index, b_index in enumerate(cand_indices):
            if index != 0 and index % 100 == 0:
                say("  {}/{}".format(index, len(cand_indices)))

            avg_cost, pred_a, pred_r = decoder.train(b_index)

            n_agents = tr_n_thread_agents[b_index]
            evaluator.update(n_agents, avg_cost, pred_a, pred_r)

        end = time.time()
        say('\n\tTime: %f' % (end - start))
        evaluator.show_results()

        ############
        # Dev data #
        ############
        if argv.dev_data:
            say('\n\n  DEV\n  ')
            predict(decoder, evaluator, dev_samples, dev_n_thread_agents)

            if evaluator.acc_t > best_acc:
                best = True
                best_epoch = epoch + 1
                best_acc = evaluator.acc_t
                decoder.output('result.dev', dev_dataset, token_dict)

                if argv.save:
                    fn = 'Model-%s.Unit-%s.batch-%d.reg-%f.psents-%d.mwords-%d' %\
                         (argv.model, argv.unit, argv.batch, argv.reg, n_prev_sents, max_n_words)
                    io_utils.dump_data(model, fn)

            say('\n\n  BEST DEV ACC: Both:%f Epoch:%d' % (best_acc, best_epoch))

        #############
        # Test data #
        #############
        if argv.test_data:
            say('\n\n\r  TEST\n  ')
            predict(decoder, evaluator, test_samples, test_n_thread_agents)
            if best:
                decoder.output('result.test', test_dataset, token_dict)
                say('\n\n  BEST TEST ACC: Both:%f Epoch:%d' % (evaluator.acc_t, best_epoch))


def predict(decoder, evaluator, samples, n_thread_agents):
    decoder.answer_a = []
    decoder.answer_r = []
    decoder.prob_r = []
    start = time.time()

    for i, sample in enumerate(zip(*samples)):
        if i != 0 and i % 100 == 0:
            say("  {}/{}".format(i, len(samples[0])))

        pred_a, pred_r = decoder.predict(sample)

        n_agents = n_thread_agents[i]
        evaluator.update(n_agents, 0., pred_a, pred_r)

    end = time.time()
    say('\n\tTime: %f' % (end - start))
    evaluator.show_results()


def main(argv):
    print '\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n'
    train(argv)
