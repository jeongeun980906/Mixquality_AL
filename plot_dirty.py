import matplotlib.pyplot as plt
import numpy as np
import os,json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str,default='mln',help='[base ,mln, mdn]')
parser.add_argument('--dataset', type=str,default='dirty_mnist',help='dataset_name')

parser.add_argument('--id', type=int,default=1,help='id')

args = parser.parse_args()
if args.mode == 'mln':
    method = ['epistemic','aleatoric','pi_entropy','maxsoftmax','entropy','bald','random']
    color = ['slateblue','forestgreen','violet','orange','cadetblue','royalblue','crimson']
elif args.mode == 'mln_sn':
    method = ['epistemic','aleatoric','pi_entropy','maxsoftmax','entropy','bald','density','random']
    color = ['slateblue','forestgreen','violet','orange','cadetblue','royalblue','slategray','crimson']
elif args.mode == 'base':
    method = ['maxsoftmax','entropy','coreset','random']
    color = ['orange','cadetblue','goldenrod','crimson']
elif args.mode == 'bald':
    method = ['maxsoftmax','entropy','bald','mean_std','random']
    color = ['orange','cadetblue','royalblue','peru','crimson']
elif args.mode == 'ddu':
    method = ['epistemic','aleatoric','random']
    color = ['slateblue','forestgreen','crimson']
else:
    raise NotImplementedError

test_acc={}
train_acc = {}
ratio = {}
for m in method:
    DIR = './res/{}_dirty_mnist_{}/{}/log.json'.format(args.mode,m,args.id)
    with open(DIR) as f:
        data = json.load(f)
    test_acc[m] = data['test_accr']
    train_acc[m] = data['train_accr']
    pool_index = data['pool_index']
    ratio[m] = list()
    for key in pool_index:
        if int(key)>0:
            temp = np.asarray(pool_index[key])
            total_len = temp.shape[0]
            temp = np.where(temp<1000)[0]
            ratio[m].append(temp.shape[0]/total_len)
    f.close()
mode = args.mode.upper()
dataset = args.dataset.capitalize()
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.title("{} Dirty MNIST Active Learning Test Accuracy".format(mode))
for m,c in zip(method,color):
    #plt.plot(test_acc[m],label=m)
    plt.plot(test_acc[m],label=m,marker='o',markersize=3,color=c)
plt.xlabel("Query Step")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks([i*2 for i in range(int(len(test_acc[method[0]])/2))])

plt.subplot(2,1,2)
plt.title("Ratio of Clean")
for e,m in enumerate(method):
    #x = [i+0.1*e for i in range(len(ratio[m]))]
    plt.plot(ratio[m],label=m,color=color[e],marker='o',markersize=3)
plt.xlabel("Query Step")
plt.ylabel("Ratio of Clean")
plt.legend(loc='upper center', ncol=5,bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig("./res/{}_dirty_mnist_{}.png".format(args.mode,args.id))
#plt.show()