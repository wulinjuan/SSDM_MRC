import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='Paraphrase using PyTorch')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type=bool, default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--save_file_path', type=str, default="result/mbert_pos",  ####### change the file name
                             help='file path to save model and log')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_file', type=str, default="../../data/SSDM_data/DP/parallel_sentence.txt",
                            help='training data from UD2.7')
    data_group.add_argument('--eval_file', type=list,
                            default=['../../data/SSDM_data/zh_sts/dev.csv', '../../data/SSDM_data/zh_sts/test.csv'],
                            help='evaluation and test file of chinese STS task')

    ######## change the config to train mrc based on different plm models
    plm_group = parser.add_argument_group('plm_config')
    plm_group.add_argument('--ml_type', type=str, default="mbert",
                           help='pretrained language model type:mbert, xlm, xlmr')
    plm_group.add_argument('--ml_token', type=str, default="bert-base-multilingual-cased",
                           help='pretrained language model token file: bert-base-multilingual-cased, xlm-roberta-base, xlm-mlm-100-1280')
    plm_group.add_argument('--ml_model_path', type=str, default="bert-base-multilingual-cased",
                           help='pretrained language model file: bert-base-multilingual-cased, xlm-roberta-base, xlm-mlm-100-1280')

    config_group = parser.add_argument_group('model_configs')
    # SSDM config
    config_group.add_argument('-m', '--margin',
                              dest='m',
                              type=float,
                              default=0.4,
                              help='margin for the training loss')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=5e-5,
                              help='learning rate')
    config_group.add_argument('-pratio', '--ploss_ratio',  # WPL
                              dest='pratio',
                              type=float,
                              default=1,
                              help='ratio of position loss')
    config_group.add_argument('-lratio', '--logloss_ratio',  # RL
                              dest='lratio',
                              type=float,
                              default=1,
                              help='ratio of reconstruction log loss')
    config_group.add_argument('-dratio', '--disc_ratio',  # language discriminative loss LDL
                              dest='dratio',
                              type=float,
                              default=0,
                              help='ratio of discriminative loss')
    config_group.add_argument('-plratio', '--para_logloss_ratio',  # CRL
                              dest='plratio',
                              type=float,
                              default=1,
                              help='ratio of paraphrase log loss')

    ####### change the parameter to choice the syntax loss
    config_group.add_argument('-posratio', '--pos_ratio',  # POS
                              dest='posratio',
                              type=float,
                              default=1,
                              help='ratio of pos loss')
    config_group.add_argument('-dpratio', '--tree_ratio',  # STL
                              dest='dpratio',
                              type=float,
                              default=0,
                              help='ratio of tree loss')

    config_group.add_argument('--eps',
                              type=float,
                              default=1e-4,
                              help='for avoiding numerical issues')
    config_group.add_argument('-edim', '--embed_dim',
                              dest='edim',
                              type=int, default=768,
                              help='size of embedding from pre-trained model')
    config_group.add_argument('-dp', '--dropout',
                              dest='dp',
                              type=float, default=0.00,
                              help='dropout probability')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=None,
                              help='gradient clipping threshold')
    # recurrent neural network detail lm_embed_size
    config_group.add_argument('-desize', '--decoder_size',
                              dest='desize',
                              type=int, default=400,
                              help='decoder hidden size')
    config_group.add_argument('--ysize',
                              dest='ysize',
                              type=int, default=200,
                              help='size of vMF')
    config_group.add_argument('--zsize',
                              dest='zsize',
                              type=int, default=200,
                              help='size of Gaussian')

    # feedforward neural network
    config_group.add_argument('-mhsize', '--mlp_hidden_size',
                              dest='mhsize',
                              type=int, default=400,
                              help='size of hidden size')
    config_group.add_argument('-mlplayer', '--mlp_n_layer',
                              dest='mlplayer',
                              type=int, default=3,
                              help='number of layer')
    config_group.add_argument('-zmlplayer', '--zmlp_n_layer',
                              dest='zmlplayer',
                              type=int, default=1,
                              help='number of layer')
    config_group.add_argument('-ymlplayer', '--ymlp_n_layer',
                              dest='ymlplayer',
                              type=int, default=1,
                              help='number of layer')

    # optimization
    config_group.add_argument('-mb', '--mega_batch',
                              dest='mb',
                              type=int, default=2,
                              help='size of mega batching')
    config_group.add_argument('-ps', '--p_scramble',
                              dest='ps',
                              type=float, default=0.,
                              help='probability of scrambling')
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')
    config_group.add_argument('-vmkl', '--max_vmf_kl_temp',
                              dest='vmkl', type=float, default=1e-2,
                              help='temperature of kl divergence')
    config_group.add_argument('-gmkl', '--max_gauss_kl_temp',
                              dest='gmkl', type=float, default=1e-3,
                              help='temperature of kl divergence')

    setup_group = parser.add_argument_group('train_setup')
    # train detail
    setup_group.add_argument('--save_dir', type=str, default=None,
                             help='model save path')
    basic_group.add_argument('--embed_type',
                             type=str, default="lm",
                             choices=['paragram', 'glove'],
                             help='types of embedding: paragram, glove')
    setup_group.add_argument('--n_epoch', type=int, default=3,
                             help='number of epochs')
    setup_group.add_argument('--batch_size', type=int, default=30,
                             help='batch size')
    setup_group.add_argument('--opt', type=str, default='adam',
                             help='types of optimizer')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=5,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=5,
                            help='evaluate model after \
                            this number of iterations')
    return parser
