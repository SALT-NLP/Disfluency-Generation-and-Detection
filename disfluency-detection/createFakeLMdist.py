import random
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse

parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument('-infile', type=str, default='news_to_fake_3m')
parser.add_argument('-model_path', type=str, default='news_32_29_whole_sample_generated_pretrain.txt')
parser.add_argument('-outfile', type=str, default='news_lm_fake_3m_trainfromswbd')
parser.add_argument('-gpu', type=str, default='2222')
parser.add_argument('-queue_size', type=int, default=20)

args = parser.parse_args()

def insert(pos, count,sent,tagSent, model,tokenizer,max_len,stop_list,device):
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

def procs(sent,tagSent,model,tokenizer,max_len,stop_list,counts,actions,wordCounts,device):
    assert (len(sent) > 0)
    count = random.choice(counts)
    allPos = range(1,len(sent))
    poses = random.choices(allPos, k=count)
    poses.sort(reverse=True)
    for pos in poses:
        #action = random.choice(actions)
        wordCount = random.choice(wordCounts)
        #if action == 1:
        insert(pos, wordCount, sent, tagSent, model,tokenizer,max_len,stop_list,device)
        '''else:
            repeat(pos, wordCount, sent, tagSent)'''
    assert sent[0]=='<bos>' and sent[-1]=='<eos>'
    sent=sent[1:-1]
    tagSent=tagSent[1:-1]
    assert (len(sent) == len(tagSent))
    return (sent,tagSent)

def read_anno(anno_path, opt):
    js_list=[]
    idx=0
    with open(anno_path, "r", encoding='utf-8') as reader:
        for line in reader:
            idx+=1
            js_list.append({'src':line.strip().split()})
            if idx>=opt.translate_num:
                break
    return js_list

def spawn(fn, args=(), nprocs=1, smp=None, daemon=False):
    if smp is None:
        smp = mp.get_context('spawn')
    assert(len(args)==6)
    sents, tags,world_size,gpus,r_list,semaphore=args
    assert len(sents)==len(tags)
    n = (len(sents) - 1) // world_size + 1
    processes = []
    for i in range(nprocs):
        process = smp.Process(
            target=fn,
            args=(i, sents[i * n:(i + 1) * n],tags[i * n:(i + 1) * n],world_size,gpus,r_list,semaphore),
            daemon=daemon,
        )
        process.start()
        processes.append(process)

    # Loop on join until it returns True or raises an exception.
    for p in processes:
        p.join()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=rank, world_size=world_size)
    torch.manual_seed(42)

def run(rank,sents,tags,world_size,gpus,r_list,semaphore):
    print('k',rank,'g', gpus[rank])
    if not rank is None:
        setup(rank,world_size)
        device = torch.device("cuda:" + str(gpus[rank]) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelWithLMHead.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    max_len = model.config.max_position_embeddings
    stop_list = tokenizer.convert_tokens_to_ids(['<eos>', '<eod>'])
    counts = [1, 2, 3]
    actions = [0, 1]
    wordCounts = [1, 2, 3, 4, 5, 6]
    for sent,tag in zip(sents,tags):
        t=procs(sent,tag,model,tokenizer,max_len,stop_list,counts,actions,wordCounts,device)
        r_list.put(t)
        semaphore.acquire()
    r_list.put(None)
    semaphore.acquire()

def consume(gpu_count, r_list, semaphore, len_sents):
    fsents = []
    ftags = []
    finished = 0
    i=0
    while True:
        if not r_list.empty():
            item=r_list.get()
            semaphore.release()
            if not item is None:
                sent, tagSent = item
                fsents.append(sent)
                ftags.append(tagSent)
                if i % 1000 == 999:
                    print(i)
                    with open(args.outfile, 'a', encoding='utf-8') as writer:
                        for sent, tagSent in zip(fsents, ftags):
                            writer.write(' '.join(sent) + '\n')
                            writer.write(' '.join(['P'] * len(sent)) + '\n')
                            writer.write(' '.join(tagSent) + '\n')
                            writer.write('\n')
                    fsents = []
                    ftags = []
                i += 1
            else:
                finished += 1
        if finished ==gpu_count:
            print('Finish decoding!')
            break
    assert(i==len_sents)


def main():
    gpus = [int(i) for i in args.gpu]
    print('gpus:', gpus)
    sents = []
    tags = []
    id = 0
    with open(args.infile, encoding='utf-8') as reader:
        for line in reader:
            if id % 100000 == 0:
                print(id)
            id += 1
            sent = line.strip().split()
            assert len(sent) > 2
            if len(sent) > 0:
                sents += [sent]
                tags += [['O'] * len(sent)]
    print('finish read')

    smp = mp.get_context('spawn')
    r_list = smp.SimpleQueue()
    semaphore = smp.Semaphore(len(gpus) * args.queue_size)
    torch.multiprocessing.freeze_support()
    # barrier = mp.Barrier(num_procs)
    p = smp.Process(target=consume, args=(len(gpus), r_list, semaphore,len(sents)))
    p.start()
    spawn(run, nprocs=len(gpus), smp=smp,
          args=[sents, tags, len(gpus), gpus, r_list, semaphore])
    p.join()


if __name__ == "__main__":
    main()





