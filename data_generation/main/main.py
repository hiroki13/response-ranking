import numpy as np

np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('-mode',  default='dg', help='Select a mode dg/ds/cr')

    # Data Creation
    parser.add_argument('--en', default=None, help='English files')
    parser.add_argument('--data', default=None, help='Data')
    parser.add_argument('--n_cands', type=int, default=2, help='Num of candidates')

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

