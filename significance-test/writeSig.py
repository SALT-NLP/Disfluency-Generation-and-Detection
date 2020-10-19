import argparse
parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument('-file', type=str, default='swbdIO/test.txt')
parser.add_argument('-pred_file', type=str, default='swbdIO/test.txt')
parser.add_argument('-out_file', type=str, default='swbdIO/test.txt')

args = parser.parse_args()

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


def writeSig(js_list,path,out_path):
    tags=[]
    with open(path, "r", encoding='utf-8') as reader:
        for line in reader:
            tags.append(line.strip().split())
    assert(len(tags)==len(js_list))
    with open(out_path, "w", encoding='utf-8') as writer:
        for tag,js in zip(tags,js_list):
            assert len(tag)==len(js['sent_tag'])
            pc1, pc2, rc1, rc2 = 0,0,0,0
            for p,g in zip(tag,js['sent_tag']):
                if p=='I' and g=='I':
                    pc1+=1
                    pc2+=1
                    rc1+=1
                    rc2+=1
                else:
                    if p=='I':
                        pc2+=1
                    if g=='I':
                        rc2+=1
            writer.write('{} {} {} {}'.format(pc1, pc2, rc1, rc2)+'\n')

js_list=read_anno(args.file)
writeSig(js_list,args.pred_file,args.out_file)