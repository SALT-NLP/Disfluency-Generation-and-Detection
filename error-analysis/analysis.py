import argparse
parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument('-file', type=str, default='swbdIO/test.txt')

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
                if l==['O']*len(l):
                    js_list.pop()
                    continue
                if l == ['I'] * len(l):
                    js_list.pop()
                    continue
                assert (len(l)>0)
                js_list[-1]['sent_tag']=l
            else:
                continue
    return js_list

def judgeRep(curDisf,i,sent):
    if curDisf==sent[i:i+len(curDisf)]:
        return True
    else:
        return False

def judgeCor(curDisf,i,sent):
    
    for w in curDisf:
        if w in sent[i:]:
            return True
    return False

js_list=read_anno(args.file)
totalDisf=0
repetitions=0
corrections=0
for item in js_list:
    sent = item['sent']
    tags = item['sent_tag']
    inDisf=0
    curDisf=[]
    for i,(w,t) in enumerate(zip(sent,tags)):
        if t=='I':
            inDisf=1
            curDisf.append(w)
        if t=='O' and inDisf==1:
            inDisf=0
            totalDisf+=1
            if judgeRep(curDisf,i,sent):
                repetitions+=1
            elif judgeCor(curDisf,i,sent):
                corrections+=1
                '''print(''curDisf)
                print(sent[:i])
                print(sent[i:])'''
            '''else:
                print(curDisf,sent[i:])'''
            curDisf=[]
    if inDisf==1:
        inDisf=0
        totalDisf+=1
        if judgeRep(curDisf,i,sent):
            repetitions+=1
        elif judgeCor(curDisf,i,sent):
            corrections+=1
        '''else:
            ssprint(curDisf,sent[i:])'''
        curDisf=[]
    if inDisf==1:
        inDisf=0
        totalDisf+=1

print('Total disf segs: ', totalDisf, ' Repetitions: ', repetitions, ' Percent: ', repetitions/totalDisf)
print('Total disf segs: ', totalDisf, ' Corrections: ', corrections, ' Percent: ', corrections/totalDisf)
print('Total disf segs: ', totalDisf, ' Deletion: ', totalDisf-corrections-repetitions, ' Percent: ', (totalDisf-corrections-repetitions)/totalDisf)




