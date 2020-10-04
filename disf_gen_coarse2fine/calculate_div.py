import sys
from nltk.translate.bleu_score import corpus_bleu
import argparse
from nltk.tokenize import word_tokenize
import string

def read_anno(anno_path):
    js_list=[]
    with open(anno_path, "r", encoding='utf-8') as reader:
        i = -1
        for line in reader:
            i += 1
            if i % 4 == 0:
                js_list.append({'sent':[token for token in line.strip().split()]})
            elif i % 4 == 1:
                continue
            if i % 4 == 2:
                l = line.strip().split()
                assert (len(l) == len(js_list[-1]['sent']))
                assert (len(l)>0)
                js_list[-1]['sent_tag']=l
            else:
                continue
    return js_list

def ngrams(n, text):
    if len(text)<n:
        return []
    padded = text
    return [' '.join(padded[i:i + n]) for i in range(len(text)-n+1)]

def frac(predngram, goldngram, length, gold_sent):
    c=0
    for pred in predngram:
        if not pred in goldngram:
            c+=1
                #print(pred,gold_sent)
    c=c/length
    return c

def eval_diversity(src,disfs):
    onegram = ngrams(1, src)
    twogram = ngrams(2, src)
    for disf in disfs:
        if len(disf) > 0:
            onegrams.append(frac(ngrams(1, disf), onegram, len(disf), src))
        else:
            assert(False)
        if len(disf) > 1:
            twograms.append(frac(ngrams(2, disf), twogram, len(disf), src))

js_list=read_anno(sys.argv[1])
onegrams=[]
twograms=[]
for js in js_list:
    sent=[]
    disfs=[]
    inDisf=0
    for w,t in zip(js['sent'],js['sent_tag']):
        if t=='I':
            if inDisf==0:
                disfs.append([])
            disfs[-1].append(w)
            inDisf=1
        else:
            inDisf=0
            sent.append(w)
    eval_diversity(sent,disfs)

c_correct = sum(onegrams)
acc = c_correct / len(onegrams)
print('{}: {} / {} = {:.2%}'.format('onegram', c_correct, len(onegrams), acc))

c_correct = sum(twograms)
acc = c_correct / len(twograms)
print('{}: {} / {} = {:.2%}'.format('twogram', c_correct, len(twograms), acc))