import random
from collections import Counter
from nltk.tokenize import word_tokenize
from multiprocessing import Pool,Manager
from functools import partial


manager = Manager()
counts=manager.list([1,2,3])
actions=manager.list([0,1])
wordCounts=manager.list([1,2,3,4,5,6])
punc_list=manager.list(['.','?','!',';',':','-','--','(',')','[',']','{','}',"'",'"','`','``',"''",'...',','])

def insert(pos, count,sent,ngrams,tagSent):
    sent[pos:pos]=random.choice(ngrams[count]).split()
    tagSent[pos:pos]=['I']*count

def repeat(pos,count,sent,tagSent):
    if pos+count>len(sent):
        count=len(sent)-pos
    sent[pos:pos]=sent[pos:pos+count]
    tagSent[pos:pos] = ['I'] * count

def procs(t):
    sent,tagSent=t
    assert (len(sent) > 0)
    count = random.choice(counts)
    allPos = range(len(sent))
    poses = random.choices(allPos, k=count)
    poses.sort(reverse=True)
    for pos in poses:
        action = random.choice(actions)
        wordCount = random.choice(wordCounts)
        if action == 1:
            insert(pos, wordCount, sent, ngrams, tagSent)
        else:
            repeat(pos, wordCount, sent, tagSent)
    assert (len(sent) == len(tagSent))
    return (sent,tagSent)

stats=[]
for i in range(7):
    stats.append(Counter())

sents=[]
tags=[]
id=0
#with open('news.2016.en.shuffled') as reader:
with open('disf_gen_coarse2fine/flt_l800.test') as reader:
    for line in reader:
        '''if id>3000000:
            break'''

        if id % 100000 == 0:
            print(id)
        id+=1
        sent=line.strip()
        sent=word_tokenize(sent)
        sent=[token.lower() for token in sent if not token in punc_list]
        if len(sent)==0:
            continue
        for i in wordCounts:
            if i==1:
                stats[i].update(sent)
            else:
                if len(sent)>=i:
                    for j in range(len(sent)-i+1):
                        stats[i][' '.join(sent[j:j+i])]+=1
        if len(sent)>0:
            sents+=[sent]
            tags+=[['O']*len(sent)]


print ('finish read')
ngrams=[[]]
for i in wordCounts:
    ngrams.append([gram for gram,_ in stats[i].most_common()])

print('finish count')

fsents=[]
ftags=[]
l1=len(sents)
l2=len(tags)
assert(l1==l2)
pool = Pool(processes=5)
i=0
'''for sent,tagSent in pool.imap(procs, zip(sents[:l1//2],tags[:l2//2]), 20000):'''
for sent,tagSent in pool.imap(procs, zip(sents,tags), 200):
    if i%1000==0:
        print(i)
    fsents.append(sent)
    ftags.append(tagSent)
    i += 1

pool.close()
pool.join()

#with open('fakeData_woPunc_news3m.txt','w') as writer:
with open('disf_gen_coarse2fine/disf_l1000.test_l800.random_gen_conll','w') as writer:
    for sent,tagSent in zip(fsents,ftags):
        writer.write(' '.join(sent)+'\n')
        writer.write(' '.join(['P']*len(sent)) + '\n')
        writer.write(' '.join(tagSent) + '\n')
        writer.write('\n')
        
with open('disf_gen_coarse2fine/disf_l800.test.random_gen','w') as writer:
    for sent,tagSent in zip(fsents,ftags):
        writer.write(' '.join(sent)+'\n')


'''with open('fakeData_woPunc_mixDisf.txt','w') as writer:
    for sent,tagSent in zip(fsents,ftags):
        writer.write(' '.join(sent)+'\n')
        writer.write(' '.join(['P']*len(sent)) + '\n')
        writer.write(' '.join(tagSent) + '\n')
        writer.write('\n')
        
with open('fakeData_woPunc_mixFlt.txt','w') as writer:
    for sent,tagSent in zip(sents[l1//2:],tags[l2//2:]):
        writer.write(' '.join(sent)+'\n')
        writer.write(' '.join(['P']*len(sent)) + '\n')
        writer.write(' '.join(tagSent) + '\n')
        writer.write('\n')'''
