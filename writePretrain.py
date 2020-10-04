import argparse

parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument('-infile', type=str, default='disf_gen_coarse2fine/swbdIO/news_32_29_whole_sample_generated')
parser.add_argument('-infile2', type=str, default='')
parser.add_argument('-infile3', type=str, default='')
parser.add_argument('-outfile', type=str, default='news_32_29_whole_sample_generated_pretrain.txt')
parser.add_argument('-strip', action='store_true')

args = parser.parse_args()

def readf(file):
    sents = []
    num=0
    with open(file,encoding='utf-8') as reader:
        i = -1
        for line in reader:
            i += 1

            if i % 4 == 0:
                t=[[token.lower()] for token in line.strip().split()]
                if t[0]==['<bos>']:
                    assert args.strip
                if args.strip:
                    assert(t[0]==['<bos>'] and t[-1]==['<eos>'])
                    t=t[:-1]
                    t=t[1:]
                sents.append(t)
            elif i % 4 == 1:
                l = line.strip().split()
                '''if not len(l) == len(sents[-1]):
                    print(len(l), len(sents[-1]))
                    num+=1'''
                if args.strip:
                    l=l[:-1]
                    l=l[1:]
                assert (len(l) == len(sents[-1]))
                for li, token in zip(sents[-1], l):
                    li.append(token)
            if i % 4 == 2:
                l = line.strip().split()
                '''if not len(l) == len(sents[-1]):
                    print(len(l), len(sents[-1]))
                    num+=1
                    print(l)
                    print(sents[-1])'''
                if args.strip:
                    l=l[:-1]
                    l=l[1:]
                assert (len(l) == len(sents[-1]))
                for li, token in zip(sents[-1], l):
                    li.append(token)
            else:
                continue
    print('n',num)
    return sents

def readf2(file):
    sents = []
    num=0
    with open(file) as reader:
        i = -1
        for line in reader:
            i += 1

            if i % 4 == 0:
                t=[[token.lower()] for token in line.strip().split()]
                sents.append(t)
            elif i % 4 == 1:
                l = line.strip().split()
                '''if not len(l) == len(sents[-1]):
                    print(len(l), len(sents[-1]))
                    num+=1'''
                assert (len(l) == len(sents[-1]))
                for li, token in zip(sents[-1], l):
                    li.append(token)
            if i % 4 == 2:
                l = line.strip().split()
                '''if not len(l) == len(sents[-1]):
                    print(len(l), len(sents[-1]))
                    num+=1
                    print(l)
                    print(sents[-1])'''
                assert (len(l) == len(sents[-1]))
                for li, token in zip(sents[-1], l):
                    li.append(token)
            else:
                continue
    print('n',num)
    return sents


def readFile(disfFile,writeFile,num,disfFile2=None,disfFile3=None):
    sents=readf(disfFile)
    if disfFile2:
        sents2 = readf2(disfFile2)
        sents3 = readf2(disfFile3)
        sents=sents+sents2+sents3
    sents = sorted(sents, key=lambda x: len(x), reverse=True)
    print('len of disf and flt:',len(sents))
    '''if num>0:
        disfSents = disfSents[:num]
        fltSents = fltSents[:num]'''
    with open(writeFile, 'w', encoding='utf-8') as writer:
        for i,sent in enumerate(sents):
            sent = [' '.join(t) for t in sent]
            writer.write(str(i)+'\t'+'\t'.join(sent)+'\n')

#readFile('fakeData_woPunc_mixDisf.txt','fakeData_woPunc_mixFlt.txt','fakeData_woPunc_pretrain_mix3m.txt',3000000)
readFile(args.infile,args.outfile,-1,disfFile2=args.infile2,disfFile3=args.infile3)