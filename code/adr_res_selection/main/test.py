from ..preprocess import convert_word_into_id, get_samples, numpy_format
from ..model import ModelAPI
from ..utils import say, load_dataset, load_init_emb, load_data


def get_datasets(argv):
    say('\nSET UP DATASET\n')

    #################
    # Load datasets #
    #################
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    say('\nLoad dataset...')
    words = load_data(argv.load_words)
    dev_dataset, _ = load_dataset(fn=argv.dev_data, vocab=words, data_size=argv.data_size, test=True)
    test_dataset, _ = load_dataset(fn=argv.test_data, vocab=words, data_size=argv.data_size, test=True)

    return dev_dataset, test_dataset, words


def create_samples(argv, dev_dataset, test_dataset, vocab_words):
    ###########################
    # Task setting parameters #
    ###########################
    n_cands = None
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    batch_size = argv.batch

    if dev_dataset:
        n_cands = len(dev_dataset[0][0][3:-1])
    elif test_dataset:
        n_cands = len(test_dataset[0][0][3:-1])
    else:
        say('\nInput dataset\n')
        exit()

    say('\n\nTASK  SETTING')
    say('\n\tResponse Candidates:%d  Contexts:%d  Max Word Num:%d' % (n_cands, n_prev_sents, max_n_words))

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
    dev_samples = numpy_format(dev_samples, batch_size, test=True)
    test_samples = numpy_format(test_samples, batch_size, test=True)

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
    dev_dataset, test_dataset, words = get_datasets(argv)
    vocab_words, init_emb = load_init_emb(argv.init_emb, words)
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
        say('\n\nDev set')
        model_api.predict_all(dev_samples)
    if test_samples:
        say('\nTest set')
        model_api.predict_all(test_samples)
    say('\n\n')
