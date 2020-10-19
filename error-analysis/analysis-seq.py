import argparse
parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument('-file', type=str, default='swbdIO/test.txt')
parser.add_argument('-pred_file', type=str, default='swbdIO/test.txt')

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

def countO(disf):
    c=0
    for i in disf:
        if not i=='I':
            c+=1
    return c


def addPred(js_list,path):
    tags=[]
    disf_js_list=[]
    with open(path, "r", encoding='utf-8') as reader:
        for line in reader:
            tags.append(line.strip().split())
    deletions=0
    repetitions=0
    corrections=0
    wrongD=0
    wrongR=0
    wrongC=0
    for tag,js in zip(tags,js_list):
        #if not len(tag)==len(js['sent_tag']):
        #print(tag,js['sent_tag'])
        assert len(tag)==len(js['sent_tag'])
        js['pred_tag']=tag
        inDisf=0
        curDisf=[]
        curPredDisf=[]
        for i,(w,t,p) in enumerate(zip(js['sent'],js['sent_tag'],tag)):
            if t=='I':
                inDisf=1
                curDisf.append(w)
                curPredDisf.append(p)
            if t=='O' and inDisf==1:
                inDisf=0
                if judgeRep(curDisf,i,js['sent']):
                    repetitions+=len(curPredDisf)
                    if not curPredDisf==['I']*len(curPredDisf):
                        wrongR+=countO(curPredDisf)

                elif judgeCor(curDisf,i,js['sent']):
                    corrections+=len(curPredDisf)
                    if not curPredDisf==['I']*len(curPredDisf):
                        wrongC+=countO(curPredDisf)
                else:
                    deletions+=len(curPredDisf)
                    if not curPredDisf==['I']*len(curPredDisf):
                        wrongD+=countO(curPredDisf)
                #print(curPredDisf)
                curDisf=[]
                curPredDisf=[]
        if inDisf==1:
            inDisf=0
            if judgeRep(curDisf,i,js['sent']):
                repetitions+=len(curPredDisf)
                if not curPredDisf==['I']*len(curPredDisf):
                    wrongR+=countO(curPredDisf)

            elif judgeCor(curDisf,i,js['sent']):
                corrections+=len(curPredDisf)
                if not curPredDisf==['I']*len(curPredDisf):
                    wrongC+=countO(curPredDisf)
            else:
                deletions+=len(curPredDisf)
                if not curPredDisf==['I']*len(curPredDisf):
                    wrongD+=countO(curPredDisf)
            curDisf=[]
            curPredDisf=[]
        if not tag==js['sent_tag']:
            disf_js_list.append(js)
    return disf_js_list, wrongD, deletions, wrongC, corrections, wrongR, repetitions



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
disf_js_list, wrongD, deletions, wrongC, corrections, wrongR, repetitions=addPred(js_list,args.pred_file)
allDisf=repetitions+corrections+deletions

print('All I: ',allDisf)

print('Wrong repetitions ', wrongR, ' Repetitions: ', repetitions, ' Percent: ', wrongR/allDisf)
print('Wrong corrections: ', wrongC, ' Corrections: ', corrections, ' Percent: ', wrongC/allDisf)
print('Wrong deletions: ', wrongD, ' Deletion: ', deletions, ' Percent: ', wrongD/allDisf)