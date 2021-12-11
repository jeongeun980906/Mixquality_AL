import torch
from core.MLN.loss import *
import matplotlib.pyplot as plt
import numpy as np

def test_eval_mln(model,data_iter,data_config,_,device):
    with torch.no_grad():
        n_total,n_correct,epis_unct_sum,alea_unct_sum,entropy_pi_sum = 0,0,0,0,0
        MI_sum = 0
        y_probs= list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            # Foraward path
            y_trgt      = batch_out.to(device)
            mln_out     = model.forward(batch_in.view(data_config[0]).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [B x N]
            mu_prime = out['mu_prime']

            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]            entropy_pi  = -pi*torch.log(pi)
            entropy_pi  = unct_out['pi_entropy']

            # BALD
            mu      = torch.softmax(mu,dim=2) #[N x K x D]
            entropy_2 = torch.sum(-mu*torch.log(mu+1e-8),dim=-1) # [N x K]
            entropy_2 = torch.sum(torch.mul(pi,entropy_2),dim=1) # [N]
            entropy_1 = torch.sum(-mu_prime*torch.log(mu_prime+1e-8),dim=-1) # [N]
            mutual_information = entropy_1-entropy_2
            entropy_pi_sum  += torch.sum(entropy_pi)
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            MI_sum += torch.sum(mutual_information)
            # Check predictions
            y_prob,y_pred    = torch.max(model_pred,1)
            n_correct   += (y_pred==y_trgt).sum().item()
            #print(y_trgt)
            n_total     += batch_in.size(0)
            
            y_probs += list(y_prob.cpu().numpy())
            
        val_accr  = (n_correct/n_total)
        entropy_pi_avg=(entropy_pi_sum/n_total).detach().cpu().item()
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        mi         = (MI_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'val_accr':val_accr,'epis':epis,'alea':alea,
                    'pi_entropy':entropy_pi_avg,'mutual_information':mi}
        model.train() # back to train mode 
    return out_eval

def func_eval_mln(model,data_iter,_,data_config,unl_size,l_size,device):
    with torch.no_grad():
        epis_unct_sum,alea_unct_sum,n_total = 0,0,0
        epis_ = list()
        alea_ = list()
        pi_entropy_ = list()
        maxsoftmax_ = list()
        entropy_ = list()
        MI_ = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in data_iter:
            # Foraward path
            mln_out     = model.forward(batch_in.view(data_config[0]).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [N x D]
            mu_prime     = out['mu_prime'] # [N x D]

            # Compute uncertainty 
            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]
            pi_entropy  = unct_out['pi_entropy']
            
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            
            y_prob,_ = torch.max(model_pred,1)
            
            entropy = torch.sum(-model_pred*torch.log(model_pred+1e-8),1)

            maxsoftmax_ += list(1-y_prob.cpu().numpy())
            
            # BALD
            mu      = torch.softmax(mu,dim=2) #[N x K x D]
            entropy_2 = torch.sum(-mu*torch.log(mu+1e-8),dim=-1) # [N x K]
            entropy_2 = torch.sum(torch.mul(pi,entropy_2),dim=1) # [N]
            entropy_1 = torch.sum(-mu_prime*torch.log(mu_prime+1e-8),dim=-1) # [N]
            mutual_information = entropy_1-entropy_2
            
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            pi_entropy_ += list(pi_entropy.cpu().numpy())
            entropy_    += list(entropy.cpu().numpy())
            MI_    += list(mutual_information.cpu().numpy())
            n_total     += batch_in.size(0)

        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'epis':epis,'alea':alea, 
                        'epis_' : epis_,'alea_' : alea_, 'maxsoftmax_':maxsoftmax_,
                        'pi_entropy_':pi_entropy_,'entropy_':entropy_,'bald_':MI_}
    return out_eval


def func_eval_mln_sn(model,gmm,data_iter,_,data_config,unl_size,l_size,device):
    with torch.no_grad():
        epis_unct_sum,alea_unct_sum,n_total = 0,0,0
        epis_ = list()
        alea_ = list()
        pi_entropy_ = list()
        maxsoftmax_ = list()
        entropy_ = list()
        feats = list()
        MI_ = list()
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in in data_iter:
            # Foraward path
            feat     = model.feature(batch_in.view(data_config[0]).to(device))
            mln_out     = model.to_logit(feat)
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out         = mln_gather(pi,mu,sigma)
            model_pred  = out['mu_sel'] # [N x D]
            mu_prime     = out['mu_prime'] # [N x D]
            # Feature Density
            feats  += [i for i in feat.detach().cpu().data] # [N x D]
            # Compute uncertainty 
            unct_out    = mln_uncertainties(pi,mu,sigma)
            epis_unct   = unct_out['epis'] # [N]
            alea_unct   = unct_out['alea'] # [N]
            pi_entropy  = unct_out['pi_entropy']
            
            epis_unct_sum += torch.sum(epis_unct)
            alea_unct_sum += torch.sum(alea_unct)
            
            y_prob,_ = torch.max(model_pred,1)
            
            entropy = torch.sum(-model_pred*torch.log(model_pred+1e-8),1)

            maxsoftmax_ += list(1-y_prob.cpu().numpy())
            
            # BALD
            mu      = torch.softmax(mu,dim=2) #[N x K x D]
            entropy_2 = torch.sum(-mu*torch.log(mu+1e-8),dim=-1) # [N x K]
            entropy_2 = torch.sum(torch.mul(pi,entropy_2),dim=1) # [N]
            entropy_1 = torch.sum(-mu_prime*torch.log(mu_prime+1e-8),dim=-1) # [N]
            mutual_information = entropy_1-entropy_2
            
            epis_ += list(epis_unct.cpu().numpy())
            alea_ += list(alea_unct.cpu().numpy())
            pi_entropy_ += list(pi_entropy.cpu().numpy())
            entropy_    += list(entropy.cpu().numpy())
            MI_    += list(mutual_information.cpu().numpy())
            n_total     += batch_in.size(0)
        
        feats = torch.stack(feats, axis=0).unsqueeze(1)
        density = gmm.q(feats).numpy()#.tolist() # [N]
        proposed_query1 = (np.log(np.exp(density/1e8)) - np.log(np.asarray(epis_))).tolist()
        proposed_query2 = (np.log(np.exp(density/1e8)) - np.log(np.asarray(MI_))).tolist()
        density_ = density.tolist()
        epis      = (epis_unct_sum/n_total).detach().cpu().item()
        alea      = (alea_unct_sum/n_total).detach().cpu().item()
        model.train() # back to train mode 
        out_eval = {'epis':epis,'alea':alea, 'density_':density_,
                        'epis_' : epis_,'alea_' : alea_, 'maxsoftmax_':maxsoftmax_,
                        'pi_entropy_':pi_entropy_,'entropy_':entropy_,'bald_':MI_,
                        'proposed_query1':proposed_query1,'proposed_query2':proposed_query2}
    return out_eval