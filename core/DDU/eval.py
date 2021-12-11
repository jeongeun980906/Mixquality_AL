import torch
'''
Aleatoric, Epistemic
'''
def func_eval_ddu(model,gmm,data_iter,_,data_config,unl_size,l_size,device):
    with torch.no_grad():
        n_total = 0
        feats = list()
        alea_ = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in data_iter:
            # Foraward path
            feat    = model.feature(batch_in.view(data_config[0]).to(device))
            model_pred = model.to_logit(feat)
            model_pred = torch.softmax(model_pred,1)

            feats  += [i for i in feat.detach().cpu().data] # [N x D]
            entropy = torch.sum(-model_pred*torch.log(model_pred+1e-8),1)
            alea_    += list(entropy.cpu().numpy())
            n_total     += batch_in.size(0)
        feats = torch.stack(feats, axis=0).unsqueeze(1)
        epis_ = gmm.q(feats).numpy().tolist() # [N]
        model.train() # back to train mode 
        out_eval = {'epis_':epis_,'alea_':alea_}
    return out_eval