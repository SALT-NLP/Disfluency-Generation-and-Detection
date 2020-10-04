import json
from collections import defaultdict

def is_eq(pred,gold):
    return pred==gold

class GenResult(object):
    def __init__(self, idx, lay, tgt,disf_frags, tags):
        self.idx = idx
        self.lay = lay
        self.tgt = tgt
        #print(tags)
        self.tgt_tags=tags
        self.gold_tgt=[]
        self.disf_frags=disf_frags
        self.correct = {}
        self.one_grams=[]
        self.two_grams=[]
        self.disflen_lessthanone=0

    def ngrams(self, n, text):
        if len(text)<n:
            return []
        padded = text
        return [' '.join(padded[i:i + n]) for i in range(len(text)-n+1)]

    def frac(self, predngram, goldngram, length, gold_sent):
        c=0
        for pred in predngram:
            if not pred in goldngram:
                c+=1
                #print(pred,gold_sent)
        c=c/length
        return c

    def eval_diversity(self,src,disfs):
        onegram = self.ngrams(1, src)
        twogram = self.ngrams(2, src)


        # print('init:',disfs,gold['sent'])

        for disf in disfs:
            # print(disfs)
            if len(disf) > 0:
                # print('ngram:',self.ngrams(1,disf),onegram)
                self.one_grams.append(self.frac(self.ngrams(1, disf), onegram, len(disf), src))
            else:
                self.disflen_lessthanone += 1

            if len(disf) > 1:
                self.two_grams.append(self.frac(self.ngrams(2, disf), twogram, len(disf), src))

    def eval(self, gold, gold_diversity=False):
        self.gold_tgt = gold['sent']
        self.gold_label = gold['sent_tag']
        if is_eq(self.lay, gold['src_label']):
            self.correct['lay'] = 1
        else:
            self.correct['lay'] = 0
        # else:
        #     print(' '.join(gold['src']))
        #     print('pred:', self.lay)
        #     print('gold:', gold['lay'])
        #     print('')

        if is_eq(self.tgt, gold['sent']):
            self.correct['tgt'] = 1
        else:
            self.correct['tgt'] = 0

        if gold_diversity:
            disfs = gold['disf_frags']
        else:
            disfs = self.disf_frags

        self.eval_diversity(gold['src'],disfs)


        # if self.correct['lay'] == 1 and self.correct['tgt'] == 1 and ('NUMBER' in self.lay and 'STRING' in self.lay and 'NAME' in self.lay):
        # if self.correct['lay'] == 1 and self.correct['tgt'] == 0:
        #     print(' '.join(gold['src']))
        #     print('pred_lay:', ' '.join(self.lay))
        #     print('gold_lay:', ' '.join(gold['lay']))
        #     print('pred_tgt:', ' '.join(self.tgt))
        #     print('gold_tgt:', ' '.join(gold['tgt']))
        #     print('')
