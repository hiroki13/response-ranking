from ..utils import say, load_dataset, load_init_emb
from model_api import ModelAPI
from preprocessor import convert_word_into_id, get_samples, theano_format


def get_datasets(argv):
    say('\nSET UP DATASET\n')

    sample_size = argv.sample_size

    #################
    # Load datasets #
    #################
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\nLoad dataset...')
    dev_dataset, word_set = load_dataset(fn=argv.dev_data, data_size=sample_size, check=argv.check)
    test_dataset, word_set = load_dataset(fn=argv.test_data, vocab=word_set, data_size=sample_size, check=argv.check)

    return dev_dataset, test_dataset, word_set


def create_samples(argv, dev_dataset, test_dataset, vocab_words):
    ###########################
    # Task setting parameters #
    ###########################
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    batch_size = argv.batch

    if dev_dataset:
        dataset = dev_dataset
    elif test_dataset:
        dataset = test_dataset
    else:
        say('\nInput dataset\n')
        exit()

    cands = dataset[0][0][3:-1]
    n_cands = len(cands)

    say('\n\nTASK  SETTING')
    say('\n\tResponse Candidates:%d  Contexts:%d  Max Word Num:%d\n' % (n_cands, n_prev_sents, max_n_words))

    ##########################
    # Convert words into ids #
    ##########################
    say('\n\nConverting words into ids...')
    # dataset: 1D: n_samples; 2D: Sample
    dev_samples = convert_word_into_id(dev_dataset, vocab_words)
    test_samples = convert_word_into_id(test_dataset, vocab_words)

    ##################
    # Create samples #
    ##################
    say('\n\nCreating samples...')
    dev_samples = get_samples(threads=dev_samples, n_prev_sents=n_prev_sents,
                              max_n_words=max_n_words, test=True)
    test_samples = get_samples(threads=test_samples, n_prev_sents=n_prev_sents,
                               max_n_words=max_n_words, test=True)

    ###################################
    # Create theano-formatted samples #
    ###################################
    dev_samples = theano_format(dev_samples, batch_size, n_cands=n_cands, test=True)
    test_samples = theano_format(test_samples, batch_size, n_cands=n_cands, test=True)

    if dev_samples:
        say('\n\nDev samples\tMini-Batch:%d' % len(dev_samples))
    if test_samples:
        say('\nTest samples\tMini-Batch:%d' % len(test_samples))

    return dev_samples, test_samples


def main(argv):
    say('\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n')

    ###############
    # Set samples #
    ###############
    dev_dataset, test_dataset, word_set = get_datasets(argv)
    vocab_words, init_emb = load_init_emb(argv.init_emb, word_set)
    dev_samples, test_samples = create_samples(argv, dev_dataset, test_dataset, vocab_words)
    del dev_dataset
    del test_dataset

    ###############
    # Set a model #
    ###############
    model_api = ModelAPI(argv, init_emb, vocab_words, argv.n_prev_sents)
    model_api.load_model()
    model_api.set_test_f()

    if dev_samples:
        say('\nDev set')
        model_api.predict_all(dev_samples)
    if test_samples:
        say('\nTest set')
        model_api.predict_all(test_samples)
    say('\n\n')
