import torch
'''
Entropy, Maxsoftmax
'''
def test_eval_baseline(model,data_iter,data_config,_,device):
    with torch.no_grad():
        n_total,n_correct,maxsoftmax_sum,entropy_sum= 0,0,0,0

        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            y_trgt      = batch_out.to(device)
            
            model_pred     = model.forward(batch_in.view(data_config[0]).to(device))
            model_pred = torch.softmax(model_pred,1)
            # Check predictions
            y_prob,y_pred    = torch.max(model_pred,1)
            n_correct   += (y_pred==y_trgt).sum().item()
            #print(y_trgt)
            n_total     += batch_in.size(0)
            
            entropy = torch.sum(-model_pred*torch.log(model_pred+1e-8),1)

            maxsoftmax_sum += torch.sum(y_prob)
            entropy_sum += torch.sum(entropy)
            
        val_accr  = (n_correct/n_total)
        maxsoftmax_avg = (maxsoftmax_sum/n_total).detach().cpu().item()
        entropy_avg = (entropy_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'val_accr':val_accr, 'maxsoftmax':maxsoftmax_avg, 'entropy':entropy_avg}
        model.train() # back to train mode 
    return out_eval

def func_eval_baseline(model,data_iter,_,data_config,unl_size,l_size,device):
    with torch.no_grad():
        n_total = 0
        maxsoftmax_ = list()
        entropy_ = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in data_iter:
            # Foraward path
            model_pred    = model.forward(batch_in.view(data_config[0]).to(device))
            model_pred = torch.softmax(model_pred,1)
            y_prob,_ = torch.max(model_pred,1)
            
            entropy = torch.sum(-model_pred*torch.log(model_pred+1e-8),1)

            maxsoftmax_ += list(1-y_prob.cpu().numpy())
            entropy_    += list(entropy.cpu().numpy())
            n_total     += batch_in.size(0)

        model.train() # back to train mode 
        out_eval = {'maxsoftmax_':maxsoftmax_,'entropy_':entropy_}
    return out_eval