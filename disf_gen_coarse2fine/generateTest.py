###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import swbddata as data

import table
import table.IO
import table.ModelConstructor
import table.Models

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
#parser.add_argument('--data', type=str, default='./data/wikitext-2',
#                    help='location of the data corpus')
#parser.add_argument('--data', type=str, default='./swbdIO',
#                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

checkpoint = torch.load(args.checkpoint,
                                map_location=lambda storage, loc: storage)
fields = checkpoint['vocab']

model_opt = checkpoint['opt']
model = table.ModelConstructor.make_base_model(
            model_opt, fields, checkpoint)

model.eval()

#corpus = data.Corpus(args.data)
ntokens = len(fields['tgt_loss'].vocab)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
q_state = table.Models.RNNDecoderState(None, None)
with open(args.outf, 'w', encoding='utf-8') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):

            inp = model.tgt_embeddings(input)

            _, q_state, _, dec_rnn_output, _ = model.tgt_decoder(
                inp, None, q_state, no_attention=model_opt.no_attention)
            output = model.tgt_classifier(dec_rnn_output, logsf=False)
            output [:,fields['tgt_loss'].vocab.stoi[table.IO.EOS_WORD]] = -float('inf')
            output[:, fields['tgt_loss'].vocab.stoi[table.IO.EOD_WORD]] = -float('inf')
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)

            word = fields['tgt_loss'].vocab.itos[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
