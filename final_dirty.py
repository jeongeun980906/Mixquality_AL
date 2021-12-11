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
ratio = {}
for a,m in zip(mode,method):
    DIR = './res/{}_dirty_mnist_{}/{}/log.json'.format(a,m,idx)
    with open(DIR) as f:
        data = json.load(f)
    test_acc[a+'_'+m] = data['test_accr']
    train_acc[a+'_'+m] = data['train_accr']
    pool_index = data['pool_index']
    ratio[a+'_'+m] = list()
    for key in pool_index:
        if int(key)>0:
            temp = np.asarray(pool_index[key])
            total_len = temp.shape[0]
            temp = np.where(temp<1000)[0]
            ratio[a+'_'+m].append(temp.shape[0]/total_len)
    f.close()

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.title("Dirty MNIST Active Learning Test Accuracy")
for a,m,c in zip(mode,method,color):
    #plt.plot(test_acc[m],label=m)
    plt.plot(test_acc[a+'_'+m],label=a+'_'+m,marker='o',markersize=3,color=c)
plt.xlabel("Query Step")
plt.ylabel("Accuracy")
plt.legend()
plt.xticks([i*2 for i in range(int(len(test_acc[mode[0]+'_'+method[0]])/2))])

plt.subplot(2,1,2)
plt.title("Ratio of Clean")
for e,(a,m) in enumerate(zip(mode,method)):
    #x = [i+0.1*e for i in range(len(ratio[m]))]
    plt.plot(ratio[a+'_'+m],label=a+'_'+m,color=color[e],marker='o',markersize=3)
plt.xlabel("Query Step")
plt.ylabel("Ratio of Clean")
plt.legend(loc='upper center', ncol=4,bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.savefig("./res/dirty_mnist_{}.png".format(idx))
#plt.show()