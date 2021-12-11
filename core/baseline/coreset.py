from numpy import dsplit, matrix
import torch

def func_eval_coreset(model,unlabel_iter,label_iter,data_config,unl_size,l_size,device):
    unlabeled = torch.zeros(unl_size,data_config[1])
    labeled = torch.zeros(l_size,data_config[1])
    
    with torch.no_grad():
        n_total = 0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in unlabel_iter:
            # Foraward path
            model_pred    = model.forward(batch_in.view(data_config[0]).to(device))
            model_pred = torch.softmax(model_pred,1)
            unlabeled[n_total:n_total+batch_in.size(0),:] = model_pred
            n_total     += batch_in.size(0)
        n_total =0
        for batch_in,_ in label_iter:
            # Foraward path
            model_pred    = model.forward(batch_in.view(data_config[0]).to(device))
            model_pred = torch.softmax(model_pred,1)
            labeled[n_total:n_total+batch_in.size(0),:] = model_pred
            n_total     += batch_in.size(0)
        model.train() # back to train mode 
    out_eval = {'unlabeled':unlabeled,'labeled':labeled}
    return out_eval

class coreset():
    def __init__(self,init_l,init_ul):
        self.init_l = init_l
        self.init_ul = init_ul
        self.key = [i for i in range(init_ul.size(0))]

    def k_center_greedy(self,query_size):
        '''
        input unlabled softmax [N x D], labled softmax [M x D]
        '''
        #_pool_size = labled.size(0)
        unlabeled = self.init_ul #[N]
        labeled = self.init_l #[M]
        output = []
        for step in range(query_size):
            distance_matrix = torch.cdist(unlabeled, labeled, p=2.0) # [N x M]
            d, _ = torch.min(distance_matrix,dim=1) # [N]
            _, current_index = torch.max(d,dim=0) # [1]
            current_index = current_index.item()
            output.append(self.key[current_index]) # return to original index
            # print(self.key[u],u,len(self.key),step)
            del self.key[current_index] # [N-step]
            labeled = torch.cat((labeled,unlabeled[current_index].unsqueeze(0)),dim=0) #[M+step x D]
            unlabeled = torch.cat((unlabeled[:current_index],unlabeled[current_index+1:]),dim=0) # [N-step x D]
            # print(labeled.size(),unlabeled.size(),step)
        output = torch.LongTensor(output)
        self.label = labeled
        self.unlabel = unlabeled
        self.key = [i for i in range(self.init_ul.size(0))]
        return output # unlabel index

    def robust_k_center(self,query_size):
        self.threshold = int(1e-4*self.init_ul.size(0))
        init_output_index = self.k_center_greedy(query_size) # [b]
        print("Done init")
        unlabeled = self.unlabel # [Ng]
        labeled = self.label # [Mg]
        distance_matrix = torch.cdist(unlabeled, labeled, p=2.0) # [Ng x Mg]
        d2_opt,_ = torch.min(distance_matrix,dim=1) # [Ng] [Ng]
        d2_opt,_ = torch.max(d2_opt,dim=0) #[1]
        lb = d2_opt.item()/2
        ub = d2_opt.item()
        delta = (lb+ub)/2
        distance_matrix = torch.cdist(self.init_ul, self.init_l, p=2.0) # [N x M]
        while abs(lb-ub)>1e-10:
            if self.check_feasible(distance_matrix,delta):
                new_distance_matrix = self.mask * distance_matrix
                print(distance_matrix.size())
                ub = torch.max(new_distance_matrix).item()
            else:
                new_distance_matrix = torch.where(self.mask==1,distance_matrix,torch.tensor(1, dtype=torch.float))
                lb = torch.min(new_distance_matrix).item()
            delta = (lb+ub)/2
            print(delta,lb,ub)
        print((self.mask==0).sum())


    def check_feasible(self,distance_matrix,delta):
        matrix = torch.ones_like(distance_matrix,dtype=torch.int64)
        matrix = torch.where(distance_matrix>delta,matrix,0)
        self.mask = matrix
        if (matrix==0).sum()<self.threshold:
            return True
        else:
            return False