from dataloader.dirtymnist import DirtyMNIST
from dataloader.ood_mnist import OODMNIST
import torch.utils.data as data
import copy
from PIL import Image
import torch

class total_dataset(data.Dataset):
    def __init__(self,dataset_name='ood_mnist',root='./dataset',train=True):
        self.dataset_name = dataset_name
        if dataset_name == 'ood_mnist':
            oodminst = OODMNIST(root, train=train,download=True)
            self.x = oodminst.data
            self.y = oodminst.targets
        elif dataset_name == 'dirty_mnist':
            dirtymnist = DirtyMNIST(root, download= True, train=train)
            self.x = dirtymnist.data
            self.y = dirtymnist.targets
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img, target = self.x[index], self.y[index]
        return img,target
    
    def __len__(self):
        return len(self.x)

class subset_dataset(data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        '''
        only used for training
        '''
        img, target = self.x[index], self.y[index]
        return img,target
        
    def __len__(self):
        return len(self.x)

class quey_dataset(data.Dataset):
    def __init__(self,x):
        self.x = x

    def __getitem__(self, index):
        '''
        only used for inference
        '''
        img = self.x[index]
        return img
        
    def __len__(self):
        return self.x.size(0)