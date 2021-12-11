import matplotlib.pyplot as plt
import numpy as np
import os,json

mode = ['mln_sn']*3 + ['bald']*2 + ['ddu'] + ['base']
method = ['epistemic','bald','density',
                'bald','mean_std','epistemic','coreset']
color = ['slateblue','slategray','royalblue','forestgreen','violet',
                    'orange','peru','crimson']#,'goldenrod']
idx = 2
test_acc={}
train_acc = {}
ratio_clean = {}
ratio_ood = {}
for a,m in zip(mode,method):
    DIR = './res/{}_ood_mnist_{}/{}/log.json'.format(a,m,idx)
    with open(DIR) as f:
        data = json.load(f)
    test_acc[a+'_'+m] = data['test_accr']
    train_acc[a+'_'+m] = data['train_accr']
    pool_index = data['pool_index']
    ratio_clean[a+'_'+m] = list()
    ratio_ood[a+'_'+m] = list()
    for key in pool_index:
        if int(key)>0:
            temp = np.asarray(pool_index[key])
            total_len = temp.shape[0]
            temp1 = np.where(temp<1000)[0]
            ratio_clean[a+'_'+m].append(temp1.shape[0]/total_len)
            temp1 = np.where(temp>61000)[0]
            ratio_ood[a+'_'+m].append(temp1.shape[0]/total_len)
    f.close()

plt.figure(figsize=(8,12))
plt.subplot(3,1,1)
plt.title("OOD MNIST Active Learning Test Accuracy")
for a,m,c in zip(mode,method,color):
    #plt.plot(test_acc[m],label=m)
    plt.plot(test_acc[a+'_'+m],label=a+'_'+m,marker='o',markersize=3,color=c)
plt.xlabel("Query Step")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks([i*2 for i in range(int(len(test_acc[mode[0]+'_'+method[0]])/2))])

plt.subplot(3,1,2)
plt.title("Ratio of Clean")
for e,(a,m) in enumerate(zip(mode,method)):
    #x = [i+0.1*e for i in range(len(ratio[m]))]
    plt.plot(ratio_clean[a+'_'+m],color=color[e],marker='o',markersize=3)
plt.xlabel("Query Step")
plt.ylabel("Ratio of Clean")
plt.tight_layout()
plt.savefig("./res/ood_mnist_{}.png".format(idx))


plt.subplot(3,1,3)
plt.title("Ratio of OOD")
for e,(a,m) in enumerate(zip(mode,method)):
    #x = [i+0.1*e for i in range(len(ratio[m]))]
    plt.plot(ratio_ood[a+'_'+m],label=a+'_'+m,color=color[e],marker='o',markersize=3)
plt.xlabel("Query Step")
plt.ylabel("Ratio of OOD")
plt.legend(loc='upper center', ncol=5,bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig("./res/ood_mnist_{}.png".format(idx))
#plt.show()