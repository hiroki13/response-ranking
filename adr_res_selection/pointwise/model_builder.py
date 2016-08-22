import theano.tensor as T
from nn.nn_utils import tanh, relu

import dynamic_model
import static_model


def set_model(argv, emb, vocab, n_prev_sents):
    #####################
    # Network variables #
    #####################
    c = T.itensor3('c')
    r = T.itensor3('r')
    a = T.ftensor3('a')
    y_r = T.ivector('y_r')
    n_r = T.ivector('n_r')
    y_a = T.ivector('y_a')
    n_a = T.ivector('n_a')
    n_agents = T.iscalar('n_agents')

    ####################
    # Model parameters #
    ####################
    opt = argv.opt
    lr = argv.lr
    init_emb = emb
    dim_emb = argv.dim_emb if emb is None else len(emb[0])
    dim_hidden = argv.dim_hidden
    n_vocab = len(vocab)
    L2_reg = argv.reg
    unit = argv.unit
    activation = relu if argv.activation == 'relu' else tanh

    #################
    # Build a model #
    #################
    print '\tMODEL: %s  Unit: %s  Opt: %s  Activation: %s' % (argv.model, unit, argv.opt, argv.activation)
    if argv.model == 'static':
        return static_model.Model(c=c, r=r, a=a, y_r=y_r, n_r=n_r, y_a=y_a, n_a=n_a, n_agents=n_agents,
                                  opt=opt, lr=lr, init_emb=init_emb,
                                  dim_emb=dim_emb, dim_hidden=dim_hidden,
                                  n_vocab=n_vocab, max_n_agents=n_prev_sents+1,
                                  L2_reg=L2_reg, unit=unit, activation=activation)
    else:
        return dynamic_model.Model(c=c, r=r, a=a, y_r=y_r, n_r=n_r, y_a=y_a, n_a=n_a, n_agents=n_agents,
                                   n_vocab=n_vocab, init_emb=init_emb, dim_emb=dim_emb, dim_hidden=dim_hidden,
                                   L2_reg=L2_reg, unit=unit, opt=opt, lr=lr, activation=activation)
