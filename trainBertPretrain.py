from utilsBertPretrain import build_vocab,batchIter,idData,idDataPretrain,isEdit,readFile
from preprocessPTBIO import readData,preProcess,readDataFromConll
import torch
import torch.nn as nn
import time
import argparse
import random
import numpy as np
from torch import optim
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup#WarmupLinearSchedule

parser = argparse.ArgumentParser(description='Disfluency Detection')
parser.add_argument("-s", "--savemodel", help="save model during evaluation", action="store_true", default=False)
parser.add_argument("-t", "--test", help="only test model", action="store_true", default=False)
parser.add_argument("-r", "--reload", help="reload model", action="store_true", default=False)
parser.add_argument("-rc", "--reload_misc", help="reload optim etc", action="store_true", default=False)
parser.add_argument("-p", "--pretrain", help="pretrain model", action="store_true", default=False)
parser.add_argument('-model', type=str, default='output_model/1.model')
parser.add_argument('-output_val', type=str, default='predictVal')
parser.add_argument('-output_test', type=str, default='predictTest')
parser.add_argument('-pretrain_data', type=str, default='news_32_29_whole_sample_generated_pretrain.txt')
parser.add_argument('-rand', type=int, default=1234, help='Random seed.')

args = parser.parse_args()

model_dir = "output_model/"
PRINT_EVERY=1000
EVALUATE_EVERY_EPOCH=1
ENCODER_LAYER=2
DROUPOUT_RATE=0.1
BATCH_SIZE=32
PRETRAIN_BATCH_SIZE=32
INIT_LEARNING_RATE=0.00001
EPOCH=20
PRETRAIN_EPOCH=10
WARM_UP_STEPS=0
ADAM_EPSILON=1e-8

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(args.rand)

class Encoder(nn.Module):
    #def __init__(self,word_size,word_dim,input_dim,hidden_dim,nLayers,labelSize,dropout_p):
    def __init__(self, labelSize):

        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, labelSize)


    def forward(self,input,lengths, train=True,hidden=None):
        encoded_layers, _ = self.bert(input)
        enc = encoded_layers##[-1]##
        output = self.fc(enc)
        output = F.log_softmax(output, dim=-1)
        return output


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt, scheduler):
        self.criterion = criterion
        self.opt = opt
        self.scheduler=None

    def __call__(self, x, y, norm,train=True):
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                #self.scheduler.step()

                self.opt.zero_grad()
        else:
            if self.opt is not None:
                self.opt.zero_grad()

        return loss.item() * norm


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx=padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone().detach()
        true_dist.fill_(self.smoothing / (self.size - 1))

        true_dist.scatter_(1, target.masked_fill(target == self.padding_idx,0).unsqueeze(1), self.confidence)
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        ret=self.criterion(x, true_dist)
        return ret

def run_epoch(data_iter, model, loss_compute,train=True,id2label=None):

    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    editTrueTotal = 0
    editPredTotal = 0
    editCorrectTotal = 0
    tagSents=[]
    for i, (sent_batch,tag_batch) in enumerate(data_iter):
        if not sent_batch[0].shape[1] == tag_batch[0].shape[1]:
            print('a',sent_batch[0].shape[1], tag_batch[0].shape[1])
        out = model(sent_batch[0], sent_batch[1],train=train)
        loss = loss_compute(out, tag_batch[0], sent_batch[2],train=train)
        total_loss += loss
        total_tokens += sent_batch[2]
        tokens += sent_batch[2]

        if i %  PRINT_EVERY== 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Tokens per Sec: %f Loss: %f " %
                    (i, tokens / elapsed , loss / sent_batch[2] ))
            start = time.time()
            tokens = 0
        if not train:
            pad=out.size(-1)
            aveLength=out.size(-2)
            _, results = torch.max(out.contiguous().view(-1, out.size(-1)), 1)
            results = results.detach().tolist()
            y = tag_batch[0].contiguous().view(-1).detach().tolist()
            for i in range(len(y)//aveLength):
                tagSents.append([id2label[item] for item,yi in zip(results[i*aveLength:(i+1)*aveLength],y[i*aveLength:(i+1)*aveLength]) if not yi==pad])

            for pred,gold in zip(results,y):
                if not gold==pad:
                    if isEdit(pred,id2label) and isEdit(gold,id2label):
                        editCorrectTotal+=1
                        editTrueTotal+=1
                        editPredTotal+=1
                    else:
                        if isEdit(pred,id2label):
                            editPredTotal += 1
                        if isEdit(gold,id2label):
                            editTrueTotal +=1
    f=0.0
    if not train:
        if not editPredTotal:
            editPredTotal = 1
            editCorrectTotal=1
        if not editCorrectTotal:
            editCorrectTotal=1
        p = editCorrectTotal / editPredTotal
        r = editCorrectTotal / editTrueTotal
        f=2 * p * r / (p + r)
        print("Edit word precision: %f recall: %f fscore: %f" % (p, r, f))
        return total_loss / total_tokens,f,tagSents

    return total_loss / total_tokens,f

def writeResults(file, tagSents,index):
    assert(len(tagSents)==len(index))
    sents=[]
    for i in range(len(tagSents)):
        sents.append([])
    for sent,i in zip(tagSents,index):
        sents[i].extend(sent)
    with open(file, 'w') as writer:
        for sent in sents:
            writer.write(' '.join(sent)+'\n')

def run(check_point,epoch,pretrain_epoch,model,pretrain_batch_size,batch_size,pretrainData,trainData,valData,testData,id2label,w_padding,test=False):
    valResult=[]
    valLabels=[]
    testResult=[]
    testLabels=[]
    criterion = LabelSmoothing(size=len(id2label), padding_idx=len(id2label), smoothing=0.0)
    if pretrainData is not None:
        t_total = (len(pretrainData[0]) // PRETRAIN_BATCH_SIZE + 1) * PRETRAIN_EPOCH
        print(t_total)
        optimizer = AdamW(model.parameters(), lr=INIT_LEARNING_RATE, eps=ADAM_EPSILON)
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARM_UP_STEPS, t_total=t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARM_UP_STEPS, num_training_steps=t_total)
        # model_opt = NoamOpt(HIDDEN_DIM, 1, WARM_UP_STEPS,
        # torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE,betas=(0.9, 0.98), eps=1e-9))
        lr_decay=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.985)

        idx = -1
        if args.reload_misc:
            idx = check_point["idx"]
            optimizer.load_state_dict(check_point["optimizer"])
            scheduler.load_state_dict(check_point["scheduler"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)  #####

        for i in range(idx + 1, pretrain_epoch):
            model.train()
            run_epoch(batchIter(pretrainData, pretrain_batch_size, w_tag_pad=w_padding, t_tag_pad=len(id2label)), model,
                      SimpleLossCompute(criterion, optimizer, scheduler), train=True)
            lr_decay.step()
            model.eval()
            print('Evaluation_val: pre epoch: %d' % (i))
            loss, f, tagSents = run_epoch(batchIter(valData, batch_size, w_tag_pad=w_padding, t_tag_pad=len(id2label)), model,
                                SimpleLossCompute(criterion, optimizer, scheduler), train=False, id2label=id2label)
            print('Loss:', loss)
            print('Evaluation_test: pre epoch: %d' % (i))
            loss, f, tagSents = run_epoch(batchIter(testData, batch_size, w_tag_pad=w_padding, t_tag_pad=len(id2label)), model,
                                SimpleLossCompute(criterion, optimizer, scheduler), train=False, id2label=id2label)
            print('Loss:', loss)
            #if i%5==4:
            if args.savemodel:
                torch.save(
                    {"idx": i, "encoder": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict()},
                    model_dir + args.model + '.' + str(i) + ".model")

    t_total = (len(trainData[0]) // BATCH_SIZE + 1) * EPOCH
    optimizer = AdamW(model.parameters(), lr=INIT_LEARNING_RATE, eps=ADAM_EPSILON)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARM_UP_STEPS, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARM_UP_STEPS, num_training_steps=t_total)
    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.985)

    if test:
        model.eval()
        print('Evaluation_val: ')
        loss, f = run_epoch(batchIter(valData, batch_size, w_tag_pad=w_padding, t_tag_pad=len(id2label)), model,
                            SimpleLossCompute(criterion, optimizer, scheduler), train=False, id2label=id2label)
        print('Loss:', loss)
        print('Evaluation_test: ')
        loss, f = run_epoch(batchIter(testData, batch_size, w_tag_pad=w_padding, t_tag_pad=len(id2label)), model,
                            SimpleLossCompute(criterion, optimizer, scheduler), train=False, id2label=id2label)
        print('Loss:', loss)
        return
    for i in range(epoch):
        model.train()
        run_epoch(batchIter(trainData,batch_size,w_tag_pad=w_padding,t_tag_pad=len(id2label)), model,
                  SimpleLossCompute( criterion, optimizer,scheduler),train=True)
        lr_decay.step()
        model.eval()
        print('Evaluation_val: epoch: %d' % (i))
        loss,f,valTag = run_epoch(batchIter(valData, batch_size, w_tag_pad=w_padding,t_tag_pad=len(id2label)), model,
                  SimpleLossCompute(criterion, optimizer,scheduler), train=False, id2label=id2label)
        print('Loss:', loss)
        if len(valResult)==0 or f >max(valResult):
            valLabels=valTag
        valResult.append(f)
        print('Evaluation_test: epoch: %d' % (i))
        loss,f,testTag=run_epoch(batchIter(testData, batch_size, w_tag_pad=w_padding,t_tag_pad=len(id2label)), model,
                                           SimpleLossCompute(criterion,  optimizer,scheduler), train=False, id2label=id2label)
        print('Loss:', loss)
        if len(testResult)==0 or f >max(testResult):
            testLabels=testTag
        testResult.append(f)
    valBest=max(valResult)
    print('ValBest epoch:', [i for i, j in enumerate(valResult) if j == valBest])
    writeResults(args.output_val,valLabels,valData[3])
    testBest = max(testResult)
    print('TestBest epoch:', [i for i, j in enumerate(testResult) if j == testBest])
    writeResults(args.output_test,testLabels,testData[3])

def initialize_from_pretrained(model, loaded_weights):
    own_state = model.state_dict()
    # Remove the 'module.' prefix since some models are saved using DataParallel object
    own_state.update({k.replace('module.', ''): v for k, v in loaded_weights.items()})
    model.load_state_dict(own_state)

'''trainSents=readDataFromConll('dps/swbd_new/train.txt')
valSents=readDataFromConll('dps/swbd_new/val.txt')
testSents=readDataFromConll('dps/swbd_new/test.txt')'''
trainSents=preProcess(readData('dps/swbd/train'))
valSents=preProcess(readData('dps/swbd/val'))
testSents=preProcess(readData('dps/swbd/test'))
id2label=['O','I']
label2id={'O':0,'I':1}
print("Read finished")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
if not args.test and args.pretrain:
    pretrainData=idDataPretrain(tokenizer,label2id,args.pretrain_data)
    print("Pretrain ID finished")
trainData=idData(tokenizer,trainSents,label2id)
valData=idData(tokenizer,valSents,label2id)
testData=idData(tokenizer,testSents,label2id)
print("ID finished")
check_point = {}
encoder = Encoder(len(id2label))

if args.reload:
    print('in')
    check_point = torch.load(model_dir + args.model + ".model",map_location='cuda:0')
    initialize_from_pretrained(encoder, check_point["encoder"])
#encoder = nn.DataParallel(encoder,device_ids=[2,3,4])
encoder = encoder.to(device)
if args.test:
    run(check_point, EPOCH, PRETRAIN_EPOCH, encoder, PRETRAIN_BATCH_SIZE, BATCH_SIZE, None, trainData, valData,
        testData, id2label, tokenizer._convert_token_to_id('[PAD]'),test=True)
else:
    if args.pretrain:
        run(check_point,EPOCH,PRETRAIN_EPOCH,encoder,PRETRAIN_BATCH_SIZE,BATCH_SIZE,
            pretrainData,trainData,valData,testData,id2label,tokenizer._convert_token_to_id('[PAD]'))
    else:
        run(check_point, EPOCH, PRETRAIN_EPOCH, encoder, PRETRAIN_BATCH_SIZE, BATCH_SIZE,
            None, trainData, valData, testData, id2label, tokenizer._convert_token_to_id('[PAD]'))