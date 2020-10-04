import os
from io import open
import torch
import table


class Corpus(object):
    def __init__(self, path,fields):
        self.dictionary = fields['tgt'].vocab
        #self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'val.txt'))
        #self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        print(path)
        with open(path, 'r', encoding="utf8") as reader:
            tgt_sents = []
            tgt_loss_sents=[]
            i = -1
            for line in reader:
                i += 1
                if i % 4 == 0:
                    words = line.strip().split()
                elif i % 4 == 1:
                    continue
                if i % 4 == 2:
                    l = line.strip().split()

                    assert (len(l) == len(words))
                    if l == ['O'] * len(l):
                        continue
                    fsent = []
                    for id in range(len(l)):
                        fsent.append(words[id])
                        if l[id] == 'I' and (
                                id == len(l) - 1 or l[id + 1] == 'O'):
                            fsent.append('<eod>')
                    tgt = []
                    tgt_loss=[]
                    for word in [table.IO.BOS_WORD]+fsent:
                        tgt.append(self.dictionary.stoi[word])
                    for word in fsent+[table.IO.EOS_WORD]:
                        tgt_loss.append(self.dictionary.stoi[word])
                    tgt_sents.append(torch.tensor(tgt).type(torch.long).view(-1,1))
                    tgt_loss_sents.append(torch.tensor(tgt_loss).type(torch.long))
                else:
                    continue

        return (tgt_sents,tgt_loss_sents)
