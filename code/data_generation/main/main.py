import numpy as np

np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('-mode',  default='dg', help='Select a mode dg/ds/cr')

    # Data Creation
    parser.add_argument('--en', default=None, help='English files')
    parser.add_argument('--fn', default=None, help='File Name')
    parser.add_argument('--data', default=None, help='Data')
    parser.add_argument('--train_data',  help='path to data')
    parser.add_argument('--dev_data',  help='path to data')
    parser.add_argument('--test_data',  help='path to data')

    parser.add_argument('--n_cands', type=int, default=2, help='Num of candidates')
    parser.add_argument('--n_prev_sents', type=int, default=5, help='prev sents')
    parser.add_argument('--max_n_words', type=int, default=20, help='maximum number of words for context/response')

    #################
    # Starting mode #
    #################
    argv = parser.parse_args()
    print
    print argv
    argv = parser.parse_args()

    if argv.mode == 'dg':
        import dataset_generator
        dataset_generator.main(argv)
    elif argv.mode == 'ds':
        import dataset_separator
        dataset_separator.main(argv)
    elif argv.mode == 'cr':
        import candidate_remover
        candidate_remover.main(argv)
    elif argv.mode =='dis':
        import distractor_generator
        distractor_generator.main(argv)
    elif argv.mode == 'sv':
        import stats_viewer
        stats_viewer.main(argv)

