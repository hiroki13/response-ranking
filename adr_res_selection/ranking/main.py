import numpy as np
import theano

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger')

    ################
    # Main options #
    ################
    parser.add_argument('-mode',  default='train', help='train/test')
    parser.add_argument('--train_data',  help='path to data')
    parser.add_argument('--dev_data',  help='path to data')
    parser.add_argument('--test_data',  help='path to data')
    parser.add_argument('--model',  default='static', help='model')
    parser.add_argument('--check',  default=False, help='check the code')

    #############################
    # Neural Network parameters #
    #############################
    parser.add_argument('--dim_emb',    type=int, default=50, help='dimension of embeddings')
    parser.add_argument('--dim_hidden', type=int, default=50, help='dimension of hidden layer')
    parser.add_argument('--loss', default='nll', help='loss')
    parser.add_argument('--unit', default='gru', help='unit')

    #######################
    # Training parameters #
    #######################
    parser.add_argument('--n_prev_sents', type=int, default=5, help='prev sents')
    parser.add_argument('--max_n_words', type=int, default=20, help='maximum number of words for context/response')
    parser.add_argument('--n_cands', type=int, default=2, help='number of candidate responses')
    parser.add_argument('--sample_size', type=int, default=10000000, help='sample size')

    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--activation', default='tanh', help='activation')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--attention', type=int, default=0, help='attention')

    parser.add_argument('--save', type=int, default=0, help='parameters to be saved or not')
    parser.add_argument('--load', default=None, help='model file to be loaded')

    #################
    # Starting mode #
    #################
    argv = parser.parse_args()
    print
    print argv
    print

    if argv.mode == 'train':
        import train
        train.main(argv)
    else:
        import test
        test.main(argv)

