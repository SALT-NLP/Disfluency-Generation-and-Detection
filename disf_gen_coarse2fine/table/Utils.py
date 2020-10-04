import torch
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device, requires_grad=False)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def sort_for_pack(input_len):
    idx_sorted, input_len_sorted = zip(
        *sorted(list(enumerate(input_len)), key=lambda x: x[1], reverse=True))
    idx_sorted, input_len_sorted = list(idx_sorted), list(input_len_sorted)
    idx_map_back = list(map(lambda x: x[0], sorted(
        list(enumerate(idx_sorted)), key=lambda x: x[1])))
    return idx_sorted, input_len_sorted, idx_map_back


def argmax(scores):
    return scores.max(scores.dim() - 1)[1]


def topk(scores, k):
    return scores.topk(k, dim = scores.dim() - 1)[1]


def add_pad(b_list, pad_index, return_tensor=True):
    max_len = max((len(b) for b in b_list))
    r_list = []
    for b in b_list:
        r_list.append(b + [pad_index] * (max_len - len(b)))
    if return_tensor:
        return torch.tensor(r_list,dtype=torch.long).to(device)
    else:
        return r_list
