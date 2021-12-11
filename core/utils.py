import os
import json

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)

class Logger():
    def __init__(self,path,init_indicies):
        self.path = path
        self.train_acc = []
        self.test_acc = []
        self.idx = {}
        self.idx[0]=init_indicies.numpy().tolist()
        self.flag=1
    
    def append(self,train_acc,test_acc,new):
        self.train_acc.append(train_acc)
        self.test_acc.append(test_acc)
        self.idx[self.flag]=new.numpy().tolist()
        self.flag+=1
        
    def save(self):
        data = {}
        with open(self.path,'w') as json_file:
            data['train_accr']=self.train_acc
            data['test_accr']=self.test_acc
            data['pool_index']= self.idx
            json.dump(data,json_file, indent=4)


def print_log_baseline(f,epoch,EPOCH,loss_avg,train_out,test_out):
    strTemp = ("epoch: [%d/%d] loss: [%.3f] train_accr:[%.4f] test_accr: [%.4f]"
                    %(epoch,EPOCH,loss_avg,train_out['val_accr'],test_out['val_accr']))
    print_n_txt(_f=f,_chars=strTemp)

    strTemp =  ("[Train] maxsoftmax avg: [%.4f] entropy avg: [%.3f]"%
        (train_out['maxsoftmax'],train_out['entropy']))
    print_n_txt(_f=f,_chars=strTemp)

    strTemp =  ("[Test] maxsoftmax avg: [%.3f] entropy avg: [%.3f]"%
            (test_out['maxsoftmax'],test_out['entropy']))
    print_n_txt(_f=f,_chars=strTemp)

def print_log_bald(f,epoch,EPOCH,loss_avg,train_out,test_out):
    strTemp = ("epoch: [%d/%d] loss: [%.3f] train_accr:[%.4f] test_accr: [%.4f]"
                    %(epoch,EPOCH,loss_avg,train_out['val_accr'],test_out['val_accr']))
    print_n_txt(_f=f,_chars=strTemp)

    strTemp =  ("[Train] maxsoftmax avg: [%.4f] entropy avg: [%.3f] mutual information avg: [%.3f] mean std avg: [%.3f]"%
        (train_out['maxsoftmax'],train_out['entropy'],train_out['bald'],train_out['mean_std']))
    print_n_txt(_f=f,_chars=strTemp)

    strTemp =  ("[Test] maxsoftmax avg: [%.3f] entropy avg: [%.3f] mutual information avg: [%.3f] mean std avg: [%.3f]"%
            (test_out['maxsoftmax'],test_out['entropy'],test_out['bald'],test_out['mean_std']))
    print_n_txt(_f=f,_chars=strTemp)