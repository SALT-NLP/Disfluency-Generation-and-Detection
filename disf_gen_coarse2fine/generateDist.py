from __future__ import division
import os
import argparse
import torch
import codecs
import glob
from nltk.tokenize import word_tokenize
import torch.distributed as dist
import torch.multiprocessing as mp

import table
import table.IO
import opts
from multiprocessing import Pool,Manager
from functools import partial

parser = argparse.ArgumentParser(description='generate.py')
opts.translate_opts(parser)
opt = parser.parse_args()

opt.dataset = opt.dataset + opt.tag_type
opt.anno = os.path.join(opt.root_dir, opt.dataset, '{}.txt'.format(opt.split))

if opt.beam_size > 0:
    opt.batch_size = 1

print('trans_opt:',vars(opt))

def spawn(fn, args=(), nprocs=1, smp=None, daemon=False):
    if smp is None:
        smp = mp.get_context('spawn')
    assert(len(args)==7)
    js_list,world_size,gpus,r_list,dummy_opt,opt,semaphore=args
    n = (len(js_list) - 1) // world_size + 1
    processes = []
    for i in range(nprocs):
        process = smp.Process(
            target=fn,
            args=(i, js_list[i * n:(i + 1) * n],world_size,gpus,r_list,dummy_opt,opt,semaphore),
            daemon=daemon,
        )
        process.start()
        processes.append(process)

    # Loop on join until it returns True or raises an exception.
    for p in processes:
        p.join()

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

def setup(rank, world_size,opt):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.master_port

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=rank, world_size=world_size)
    torch.manual_seed(42)

def run(rank,js_list,world_size,gpus,r_list,dummy_opt,opt,semaphore):
    print('k',rank,'g', gpus[rank])
    if not rank is None:
        setup(rank,world_size,opt)
        device = torch.device("cuda:" + str(gpus[rank]) if torch.cuda.is_available() else "cpu")
        '''n=(len(js_list)-1)//world_size+1
        js_list=js_list[rank*n:(rank+1)*n]'''
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = table.Translator(opt, dummy_opt.__dict__, device=device, parallel=True, gpu_id=gpus[rank])
    print('Finish constructing translator')
    data = table.IO.TableDataset(
        js_list, translator.fields, translator.model_opt, only_generate=True)
    print('Total number of sents in dataset:', len(data))
    test_data = table.IO.OrderedIterator(
        dataset=data, device=device, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)
    print('Total number of batches in dataset:', len(data) // opt.batch_size)

    t_list=[]
    # inference
    idx = 0
    with torch.no_grad():
        for batch in test_data:
            idx += 1
            if idx % 5000 == 0:
                print('G:', gpus[rank],'Batch:', idx)
            r = translator.translate(batch, js_list)
            for item in r:
                #print(item)
                r_list.put(item)
                semaphore.acquire()
                '''t_list.append(item)
    for item in t_list:
        r_list'''
    r_list.put(None)
    semaphore.acquire()

def consume(js_list,nprocess,r_list,opt,semaphore):
    f_list = []
    finished=0
    while True:
        if not r_list.empty():
            item=r_list.get()
            semaphore.release()
            if not item is None:
                f_list.append(item)
                '''i =0
                while True:
                    if i<len(f_list)  and f_list[i].idx<item.idx:
                        i+=1
                        continue
                    else:
                        break
                f_list[i:i]=[item]'''
            else:
                finished+=1
        if finished==nprocess:
            print('Finish decoding!')
            break
    assert len(f_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
        len(f_list), len(js_list))
    #f_list.sort(key=lambda x: x.idx)

    # evaluation
    if opt.eval_diversity:
        for pred, gold in zip(r_list, js_list):
            pred.eval_diversity(gold['src'], pred.disf_frags)
        print('Results:')
        print('disf_less_than_one:', sum((x.disflen_lessthanone for x in r_list)))
        onegrams = []
        twograms = []
        for x in f_list:
            onegrams.extend(x.one_grams)
            twograms.extend(x.two_grams)
        c_correct = sum(onegrams)
        acc = c_correct / len(onegrams)
        print('{}: {} / {} = {:.2%}'.format('onegram',
                                            c_correct, len(onegrams), acc))
        c_correct = sum(twograms)
        acc = c_correct / len(twograms)
        print('{}: {} / {} = {:.2%}'.format('twogram',
                                            c_correct, len(twograms), acc))
    assert f_list is not None
    print("Writing to " + os.path.join(opt.root_dir, opt.dataset, opt.output_file + '_generated'))
    disf_generated = 0
    with open(os.path.join(opt.root_dir, opt.dataset, opt.output_file + '_generated'), 'w', encoding='utf-8') as writer:
        for x in f_list:
            if 'I' in x.tgt_tags:
                disf_generated += 1
                assert (len(x.tgt) == len(x.tgt_tags))
                writer.write('\t'.join(x.tgt) + '\n')
                writer.write('\t'.join(['P'] * len(x.tgt_tags)) + '\n')
                writer.write('\t'.join(x.tgt_tags) + '\n')
                writer.write('\n')
    print('Disf_sents/Total_sents', disf_generated, len(f_list))


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    gpus = [int(i) for i in opt.gpu]
    print('gpus:',gpus)

    js_list = read_anno(opt.anno, opt)
    print('Finished reading, data_len',len(js_list))
    for fn_model in glob.glob(opt.model_path):
        opt.model = fn_model
        print(fn_model)
        print(opt.anno)

        smp = mp.get_context('spawn')
        r_list = smp.SimpleQueue()
        semaphore = smp.Semaphore(len(gpus) * opt.queue_size)
        torch.multiprocessing.freeze_support()
        #barrier = mp.Barrier(num_procs)
        p = smp.Process(target=consume, args=(js_list, len(gpus), r_list,opt,semaphore))
        p.start()
        spawn(run, nprocs=len(gpus), smp=smp,
                 args=[js_list, len(gpus), gpus, r_list, dummy_opt, opt, semaphore])
        p.join()






if __name__ == "__main__":
    main()
