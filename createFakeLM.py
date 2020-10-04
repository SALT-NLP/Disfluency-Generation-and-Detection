import random
from collections import Counter
from nltk.tokenize import word_tokenize
from multiprocessing import Pool,Manager
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer


import argparse

parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument('-infile', type=str, default='news_to_fake_3m')
parser.add_argument('-model_path', type=str, default='news_32_29_whole_sample_generated_pretrain.txt')
parser.add_argument('-outfile', type=str, default='news_lm_fake_3m_trainfromswbd')

args = parser.parse_args()

model = AutoModelWithLMHead.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
max_len=model.config.max_position_embeddings
stop_list=tokenizer.convert_tokens_to_ids(['<eos>','<eod>'])

'''ids= tokenizer.convert_tokens_to_ids(['<bos>','<eos>','<eod>'])
print(ids)
print(tokenizer.convert_ids_to_tokens(ids))'''

counts=[1,2,3]
actions=[0,1]
wordCounts=[1,2,3,4,5,6]

def insert(pos, count,sent,tagSent):
    prime=sent[:pos]
    encoded_prompt = tokenizer.encode(prime, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)
    if max_len-len(encoded_prompt[0])<count:
        encoded_prompt=encoded_prompt[:,-(max_len-count):]

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=count + len(encoded_prompt[0]),
        temperature=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
    )
    output_sequence=output_sequences[0].tolist()[len(encoded_prompt[0]):]
    stop_id=-1
    for i,t in enumerate(output_sequence):
        if t in stop_list:
            stop_id=i
            break
    if stop_id>=0:
        output_sequence = output_sequence[:stop_id]
    text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    insert_list=text.strip().split()
    sent[pos:pos]=insert_list
    tagSent[pos:pos]=['I']*len(insert_list)

def repeat(pos,count,sent,tagSent):
    if pos+count>len(sent):
        count=len(sent)-pos
    sent[pos:pos]=sent[pos:pos+count]
    tagSent[pos:pos] = ['I'] * count

def procs(t):
    sent,tagSent=t
    assert (len(sent) > 0)
    count = random.choice(counts)
    allPos = range(1,len(sent))
    poses = random.choices(allPos, k=count)
    poses.sort(reverse=True)
    for pos in poses:
        #action = random.choice(actions)
        wordCount = random.choice(wordCounts)
        #if action == 1:
        insert(pos, wordCount, sent, tagSent)
        '''else:
            repeat(pos, wordCount, sent, tagSent)'''
    assert (len(sent) == len(tagSent))
    return (sent,tagSent)

sents=[]
tags=[]
id=0

with open(args.infile, encoding='utf-8') as reader:
    for line in reader:
        if id % 100000 == 0:
            print(id)
        id+=1
        sent=line.strip().split()
        assert len(sent)>2
        if len(sent)>0:
            sents+=[sent]
            tags+=[['O']*len(sent)]


print ('finish read')

fsents=[]
ftags=[]
l1=len(sents)
l2=len(tags)
assert(l1==l2)
i=0
for t in zip(sents,tags):
    sent,tagSent=procs(t)

    fsents.append(sent)
    ftags.append(tagSent)
    if i%1000==999:
        print(i)
        with open(args.outfile, 'a', encoding='utf-8') as writer:
            for sent, tagSent in zip(fsents, ftags):
                writer.write(' '.join(sent) + '\n')
                writer.write(' '.join(['P'] * len(sent)) + '\n')
                writer.write(' '.join(tagSent) + '\n')
                writer.write('\n')
        fsents=[]
        ftags=[]
    i += 1





