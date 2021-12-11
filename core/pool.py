from numpy import dtype, index_exp
from torch.utils import data
from core.data import total_dataset,subset_dataset,quey_dataset
import torch
import copy
import random

class AL_pool():
    def __init__(self,root='../dataset',dataset_name='mnist',num_init=100):
        self.basedata=total_dataset(dataset_name, root=root)
        self.batch_size=128
        self.total_size = self.basedata.__len__()
        self.idx = torch.tensor(random.sample(range(1000), num_init))
        self.dataset = dataset_name
        
    def subset_dataset(self,indices):
        indices = torch.cat((self.idx,indices),0)
        self.idx,_ = indices.sort()
        x = copy.deepcopy(self.basedata.x[self.idx])
        y = copy.deepcopy(self.basedata.y[self.idx])
        total = torch.range(0,self.total_size-1,dtype=torch.int64)
        mask = torch.ones_like(total, dtype=torch.bool)
        mask[self.idx] = False
        # if torch.where(mask==False)[0].size(0)%10 !=0:
        #     print(self.idx.numpy().tolist())
        self.unlabled_idx = total[mask]
        labeled_subset = subset_dataset(x,y)
        train_loader = torch.utils.data.DataLoader(labeled_subset, batch_size=self.batch_size, 
                        shuffle=False)
        infer_loader = self.get_unlabled_pool()
        return train_loader,infer_loader

    def get_unlabled_pool(self):
        print(self.unlabled_idx.size())
        x = copy.deepcopy(self.basedata.x[self.unlabled_idx])
        query_pool = quey_dataset(x)
        loader  = torch.utils.data.DataLoader(query_pool, batch_size=self.batch_size, 
                        shuffle=False)
        return loader
