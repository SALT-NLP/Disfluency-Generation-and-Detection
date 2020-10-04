import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.new_empty((embed.weight.size(0), 1),requires_grad=True).bernoulli_(
            1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(
            masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = F.embedding(words, masked_embed_weight,
                                       padding_idx, embed.max_norm, embed.norm_type,
                                       embed.scale_grad_by_freq, embed.sparse
                                       )
    return X


if __name__ == '__main__':
    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    embed = torch.nn.Embedding(V, h)
    print('e',embed)

    words = np.random.randint(
        low=0, high=V - 1, size=(batch_size, bptt))
    words = torch.LongTensor(words)
    words = Variable(words)
    print('w',words)

    origX = embed(words)
    X = embedded_dropout(embed, words)

    print(origX)
    print(X)
