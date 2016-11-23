import tfidf

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Addressee and Response Selection System')

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
    parser.add_argument('--data_size', type=int, default=10000000, help='data size')
    parser.add_argument('--sample_size', type=int, default=1, help='sample size')

    #################
    # Starting mode #
    #################
    argv = parser.parse_args()
    print
    print argv
    print

    tfidf.main(argv)

