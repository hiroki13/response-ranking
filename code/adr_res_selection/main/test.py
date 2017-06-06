from ..preprocess import convert_word_into_id, get_samples, numpy_format
from ..model import ModelAPI
from ..utils import say, load_dataset, load_init_emb, load_multi_ling_init_emb


def get_datasets(argv):
    say('\nSET UP DATASET\n')
    say('\nLoad dataset...')
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    test_dataset, _ = load_dataset(fn=argv.test_data, vocab=None, data_size=argv.data_size, test=True)
    return test_dataset


def create_samples(argv, test_dataset, vocab_words):
    ###########################
    # Task setting parameters #
    ###########################
    n_cands = None
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    batch_size = argv.batch

    if test_dataset:
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
    test_samples = convert_word_into_id(test_dataset, vocab_words)

    ##################
    # Create samples #
    ##################
    say('\n\nCreating samples...')
    test_samples = get_samples(threads=test_samples, n_prev_sents=n_prev_sents,
                               max_n_words=max_n_words, test=True)
    test_samples = numpy_format(test_samples, batch_size, test=True)

    if test_samples:
        say('\nTest samples\tMini-Batch:%d' % len(test_samples))

    return test_samples


def main(argv):
    say('\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n')

    ###############
    # Set dataset #
    ###############
    test_dataset = get_datasets(argv)

    ##########################
    # Set initial embeddings #
    ##########################
    if argv.emb_type == 'mono':
        vocab_words, init_emb = load_init_emb(argv.init_emb)
    else:
        vocab_words, init_emb = load_multi_ling_init_emb(argv.init_emb, argv.lang)

    ###############
    # Set samples #
    ###############
    test_samples = create_samples(argv, test_dataset, vocab_words)
    del test_dataset

    ###############
    # Set a model #
    ###############
    model_api = ModelAPI(argv, init_emb, vocab_words, argv.n_prev_sents)
    model_api.set_model()
    model_api.load_params()
    model_api.set_test_f()

    say('\nTest set')
    model_api.predict_all(test_samples)
    say('\n\n')
