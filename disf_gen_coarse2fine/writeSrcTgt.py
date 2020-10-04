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

js=read_anno('swbdIO/test.txt')
with open('swbdIO/flt.test','w',encoding='utf-8') as s_writer, open('swbdIO/disf.test','w',encoding='utf-8') as t_writer:
    for item in js:
        src=[]
        tgt=[]
        for w,t in zip(item['sent'],item['sent_tag']):
            tgt.append(w)
            if t=='O':
                src.append(w)
        s_writer.write(' '.join(src)+'\n')
        t_writer.write(' '.join(tgt)+'\n')



