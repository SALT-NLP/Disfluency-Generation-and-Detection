import argparse

def preprocess_opts(parser):
    parser.add_argument('-root_dir', default='',
                        help="Path to the root directory.")
    parser.add_argument('-dataset', default='swbd',
                        help="Name of dataset.")
    parser.add_argument('-tag_type', default='IO',
                        help="Type of tag system")
    parser.add_argument('-disf_seg', action='store_true',
                        help="Conduct disf seg task")
    parser.add_argument('-no_disf_trans', action='store_true',
                        help="No disf trans task")
    parser.add_argument('-train_data', type=str, default='train.txt',
                        help='Name of training set.')
    parser.add_argument('-train_data_pt', type=str, default='train.pt',
                        help='Name of training set.')
    parser.add_argument('-bert', action='store_true',
                        help="Conduct disf seg task")
    # Dictionary Options
    parser.add_argument('-src_words_min_frequency', type=int, default=0)
    parser.add_argument('-tgt_words_min_frequency', type=int, default=0)
    parser.add_argument('-seed', type=int, default=123,
                        help="Random seed")

    # Truncation options
    parser.add_argument('-src_seq_length', type=int, default=1000,
                        help="Maximum source sequence length")
    parser.add_argument('-tgt_seq_length', type=int, default=1000,
                        help="Maximum target sequence length to keep.")

    # Data processing options
    parser.add_argument('-lower', action='store_false', help='lowercase data')
    parser.add_argument('-include_flt', action='store_true', help='include fluent sentences during generation')
    parser.add_argument('-decoder_word_input', action="store_true",
                        help='Whether to use initial word as input for tgt decoding')
    parser.add_argument('-no_connection_decoder', action="store_true",
                        help='disf label weight')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """
    # Model options
    # Embedding Options
    parser.add_argument('-word_vec_size', type=int, default=300, #250
                        help='Word embedding for both.')


    parser.add_argument('-decoder_input_size', type=int, default=300,  #200,
                        help='Layout embedding size.')

    parser.add_argument('-copy_attn_force', action="store_true",
                        help="For copiable words, only use copy score")

    parser.add_argument('-no_copy', action="store_true",
                        help="No copy mechanism")

    # RNN Options
    parser.add_argument('-seprate_encoder', action="store_true",
                        help="Use different encoders for layout and target decoding.")
    parser.add_argument('-encoder_type', type=str, default='brnn',
                        choices=['rnn', 'brnn','transformer'],
                        help="""Type of encoder layer to use.""")
    parser.add_argument('-decoder_type', type=str, default='rnn',
                        choices=['rnn','transformer'],
                        help='Type of decoder layer to use.')
    parser.add_argument('-transformer_layers', type=int, default=6,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-transformer_heads', type=int, default=8,
                        help='Number of heads in enc/dec.')
    parser.add_argument('-transformer_dim', type=int, default=512,
                        help='D_model in enc/dec.')
    parser.add_argument('-transformer_fw_dim', type=int, default=2048,
                        help='D_model in fw layer of enc/dec.')
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in enc/dec.')
    parser.add_argument('-enc_layers', type=int, default=1,
                        help='Number of layers in the encoder')
    parser.add_argument('-dec_layers', type=int, default=1,
                        help='Number of layers in the decoder')

    parser.add_argument('-rnn_size', type=int, default=300,
                        help='Size of LSTM hidden states')

    parser.add_argument('-rnn_type', type=str, default='LSTM',
                        choices=['LSTM', 'GRU'],
                        help="""The gate type to use in the RNNs""")
    parser.add_argument('-brnn_merge', default='concat',
                        choices=['concat', 'sum'],
                        help="Merge action for the bidir hidden states")

    parser.add_argument('-gelu', action="store_true",
                        help='If GELU')

    # Attention options
    parser.add_argument('-global_attention', type=str, default='general',
                        choices=['dot', 'general', 'mlp'],
                        help="""The attention type to use:
                        dotprot or general (Luong) or MLP (Bahdanau)""")
    parser.add_argument('-attn_hidden', type=int, default=-1,
                        help="if attn_hidden > 0, then attention score = f(Ue) B f(Ud)")

    # Target options
    parser.add_argument('-no_share_emb_layout_encoder', action="store_true",
                        help='Whether share embeddings for layout encoder.')

    parser.add_argument('-encode_one_pass', action="store_true",
                        help='Q and layout encode one pass')
    parser.add_argument('-copy_prb', type=str, default='hidden_context',
                        choices=['hidden', 'hidden_context'],
                        help="How to compute prb(copy). hidden: rnn hidden vector; hidden_context: also use attention context vector;")
    # Ablation
    parser.add_argument('-no_lay_encoder', action="store_true",
                        help='No layout RNN encoder.')

def train_opts(parser):
    # Model loading/saving options
    parser.add_argument('-root_dir', default='',
                        help="Path to the root directory.")
    parser.add_argument('-dataset', default='swbd',
                        help="Name of dataset.")
    parser.add_argument('-tag_type', default='IO',
                        help="Type of tag system")
    parser.add_argument('-train_data_pt', type=str, default='train.pt',
                        help='Name of training set.')
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    parser.add_argument('-train_from_pretrain', action="store_true",
                        help='Whether use disf segment loss only')
    parser.add_argument('-only_disf_loss', action="store_true",
                        help='Whether use disf segment loss only')

    # GPU

    parser.add_argument('-gpuid', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=123,
                        help="""Random seed used for the experiments
                        reproducibility.""")

    # Init options
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    #parser.add_argument('-param_init', type=float, default=0.08,
    #                    help="""Parameters are initialized over uniform distribution
    #                    with support (-param_init, param_init).
    #                    Use 0 to not use initialization""")

    # Optimization options
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('-optim', default='adam',
                        choices=['sgd', 'adagrad',
                                 'adadelta', 'adam', 'rmsprop'],
                        help="""Optimization method.""")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to
                        max_grad_norm""")
    parser.add_argument('-transformer_dropout', type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument('-dropout', type=float, default=0.5,
                        help="Dropout rate.")
    parser.add_argument('-dropout_i', type=float, default=0.5,
                        help="Dropout rate (for RNN input).")
    parser.add_argument('-lock_dropout', action='store_true',
                        help="Use the same dropout mask for RNNs.")
    parser.add_argument('-weight_dropout', type=float, default=0,
                        help=">0: Weight dropout probability; applied in LSTM stacks.")
    parser.add_argument('-dropword_enc', type=float, default=0,
                        help="Drop word rate.")
    parser.add_argument('-dropword_dec', type=float, default=0,
                        help="Drop word rate.")
    parser.add_argument('-smooth_eps', type=float, default=0,
                        help="Label smoothing")
    parser.add_argument('-moving_avg', type=float, default=0,
                        help="Exponential moving average")
    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.0005,
                        help="""Starting learning rate.""")
    parser.add_argument('-transformer_learning_rate', type=float, default=0.0001,
                        help="""Starting learning rate.""")
    parser.add_argument('-warm_up', action="store_true",
                        help='Use warm up')
    parser.add_argument('-warm_up_step', type=int, default=400,
                        help="warm_up_step")
    parser.add_argument('-warm_up_factor', type=float, default=1.0,
                        help="warm_up_factor")
    parser.add_argument('-alpha', type=float, default=0.95,
                        help="Optimization hyperparameter")
    parser.add_argument('-learning_rate_decay', type=float, default=0.985,
                        help="""If update_learning_rate, decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=0,
                        help="""Start decaying every epoch after and including this epoch""")
    parser.add_argument('-start_checkpoint_at', type=int, default=0,
                        help="""Start checkpointing every epoch after and including this epoch""")

    parser.add_argument('-report_every', type=int, default=100,
                        help="Print stats at this interval.")
    parser.add_argument('-exp', type=str, default="",
                        help="Name of the experiment for logging.")

    # loss
    parser.add_argument('-disf_label_weight', type=float, default=1.0,
                        help='disf label weight')

    parser.add_argument('-disf_gen_weight', type=float, default=1.0,
                        help='disf label weight')

    parser.add_argument('-joint_loss',  action="store_true",
                        help='disf label weight')

    parser.add_argument('-trans_pretrain_epoch', type=int, default=3,
                        help='disf label weight')

    parser.add_argument('-layout_weight', type=float, default=1.0,
                        help='disf label weight')

    parser.add_argument('-start_layout_loss', type=int, default=0,
                        help='start layout loss')

    parser.add_argument('-stop_layout_loss', type=int, default=1000,
                        help='start layout loss')

    parser.add_argument('-no_connection_encdec', action="store_true",
                        help='disf label weight')

    parser.add_argument('-no_attention', action="store_true",
                        help='disf label weight')


    #parser.add_argument('-coverage_loss', type=float, default=0,
                        #help="Attention coverage loss.")


def translate_opts(parser):
    parser.add_argument('-root_dir', default='',
                        help="Path to the root directory.")
    parser.add_argument('-dataset', default='swbd',
                        help="Name of dataset.")
    parser.add_argument('-tag_type', default='IO',
                        help="Type of tag system")
    parser.add_argument('-model_path', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-split', default="test",
                        help="Path to the evaluation annotated data")
    #parser.add_argument('-output', default='pred.txt',
                        #help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('-run_from', type=int, default=0,
                        help='Only evaluate run.* >= run_from.')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-beam_size', type=int, default=0,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help='N-best size')
    '''parser.add_argument('-max_lay_len', type=int, default=50,
                        help='Maximum layout decoding length.')
    parser.add_argument('-max_tgt_len', type=int, default=100,
                        help='Maximum tgt decoding length.')'''
    parser.add_argument('-max_disf_len', type=int, default=8,
                        help='Maximum layout decoding length.')
    parser.add_argument('-gpu', type=str, default='0',
                        help="Device to run on")
    parser.add_argument('-gold_layout', action='store_true',
                        help="Given the golden layout sequences for evaluation.")

    parser.add_argument('-random_layout', action='store_true',
                        help="Use random layout")

    parser.add_argument('-flt_gen', action='store_true',
                        help="Regenerate fluent part")

    parser.add_argument('-gold_diversity', action='store_true',
                        help="Report gold diversity")

    parser.add_argument('-no_in_sent_word', action='store_true',
                        help="no_in_sent_word")

    parser.add_argument('-random_choose_topk', action='store_true',
                        help="random_choose_topk")

    parser.add_argument('-random_sample', action='store_true',
                        help="random_sample")

    parser.add_argument('-eval_diversity', action='store_true',
                        help="evaluate diversity")

    parser.add_argument('-gen_eod', action='store_true',
                        help="generate eod")

    parser.add_argument('-attn_ignore_small', type=float, default=0,
                        help="Ignore small attention scores.")
    parser.add_argument('-include_flt', action='store_true', help='include fluent sentences during generation')
    parser.add_argument('-sample_num', type=int, default=1,
                        help='Number of samples in each step')

    parser.add_argument('-translate_num', type=int, default=0,
                        help='Number of translation sentences')

    parser.add_argument('-queue_size', type=int, default=50,
                        help='Number of translation sentences')

    parser.add_argument('-temperature', type=float, default=1.0,
                        help='temperature of flatting logits')

    parser.add_argument('-random_mask_eod', type=float, default=0.0,
                        help="""During generation, mask EOD with thos prob""")

    parser.add_argument('-output_file', default='pred.txt',
                        help="""Path to output the predictions and score (each line will be the decoded sequence""")

    parser.add_argument('-master_port', default='12355',
                        help="""Master Port""")

