import sys
from nltk.translate.bleu_score import corpus_bleu
import argparse
from nltk.tokenize import word_tokenize
import string

reference0 = open(sys.argv[2], 'r', encoding='utf-8').readlines()
#candidate = open(p+'informal', 'r').readlines()
candidate = open(sys.argv[1],'r', encoding='utf-8').readlines()


references = []
candidates = []
for i in range(len(candidate)):
    '''references.append([word_tokenize(reference0[i].strip()), word_tokenize(reference1[i].strip()),
                       word_tokenize(reference2[i].strip()), word_tokenize(reference3[i].strip())])
    candidates.append(word_tokenize(candidate[i].strip()))'''
    references.append([reference0[i].strip().split()])
    candidates.append(candidate[i].strip().split())
score = corpus_bleu(references, candidates)
print("The bleu score is: "+str(score))