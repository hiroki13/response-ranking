import theano

from ..utils import say, load_dataset, load_init_emb, load_data
from preprocessor import get_samples, theano_format


def test(argv):
    say('\nSET UP TESTING SETTINGS\n')

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
    dev_data, vocab_words = load_dataset(argv.dev_data, argv.check)
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

    dev_data = test_data = None

    #################
    # Build a model #
    #################
    say('\n\nBuilding a model...')
    model = load_data(argv.load)
    pred_f = theano.function(inputs=model.inputs,
                             outputs=[model.a_hat, model.r_hat],
                             on_unused_input='ignore'
                             )
    pred_r_f = theano.function(inputs=model.inputs,
                               outputs=model.r_hat,
                               on_unused_input='ignore'
                               )

    ############
    # Dev data #
    ############
    say('\nTesting start\n')
    if argv.dev_data:
        say('\n\n  DEV\n  ')
        dev_acc_both = model.evaluate_all(dev_samples, pred_f, pred_r_f)

    #############
    # Test data #
    #############
    if argv.test_data:
        say('\n\n\r  TEST\n  ')
        test_acc_both = model.evaluate_all(test_samples, pred_f, pred_r_f)

    say('\n\n')


def main(argv):
    print '\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n'
    test(argv)
