import pickle
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
#from nxviz.plots import CircosPlot
plt.rcParams["font.family"] = "Times New Roman"

BEH_LABELS = ['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand']
#from hmmlearn import hmm

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score as acc

def get_transitions(seq):
    X = np.zeros((9,9))
    for k in range(1,len(seq)):
        X[seq[k-1], seq[k]] += 1
    return X

def get_motif_probabilities(trans_mat, X, cut_off = 0.01):

    start_probs = X.sum(axis=1)/X.sum()
    two_motifs = []
    two_probs = []
    for i in range(len(BEH_LABELS)):
        for j in range(len(BEH_LABELS)):
            if (start_probs[i]*trans_mat[i][j] >= cut_off):# and (i not in [2,4,6]):
                two_motifs.append('{}{}'.format(i,j))
                two_probs.append(start_probs[i]*trans_mat[i][j])

    three_motifs = []
    three_probs = []
    for i in range(len(two_motifs)):
        start_pos = int(two_motifs[i][-1])
        for j in range(len(BEH_LABELS)):
            if two_probs[i]*trans_mat[start_pos, j] >= cut_off:
                three_motifs.append('{}{}'.format(two_motifs[i],j))
                three_probs.append(two_probs[i]*trans_mat[start_pos, j]) 

    two_motifs.extend(three_motifs)
    two_probs.extend(three_probs)
    return two_motifs, two_probs

'''
def count_raw_tokens(tokens):
    for i in range(len(BEH_LABELS)):
        for j in range(len(BEH_LABELS)):
'''        
def get_trans_mat_combined(datasets):
    trans_mat = np.zeros((9,9))
    for dset in datasets:
        X = pickle.load(open(dset,'rb'))
        for seq in X:
            tmat = get_transitions(seq)
            trans_mat += tmat
    rsum = np.expand_dims(np.sum(trans_mat, axis=1)+1e-15, axis=1)
    norm_trans_mat = trans_mat / rsum
    return norm_trans_mat


def get_trans_mat(dataset):
    trans_mat = np.zeros((9,9))
    X = pickle.load(open(dataset,'rb'))
    for seq in X:
        tmat = get_transitions(seq)
        trans_mat += tmat
    rsum = np.expand_dims(np.sum(trans_mat, axis=1)+1e-15, axis=1)
    norm_trans_mat = trans_mat / rsum
    return norm_trans_mat

def fit_model(hi_cat, low_cat):
    X = pickle.load(open(hi_cat,'rb'))
    pos_samples = []
    for seq in X:
        tmat = get_transitions(seq)
        rsum = np.expand_dims(np.sum(tmat, axis=1)+1e-15, axis=1)
        tmat = tmat/rsum
        pos_samples.append(tmat.flatten())
    pos_samples = np.vstack(pos_samples)
    pos_labels = np.ones((pos_samples.shape[0],))
 
    X = pickle.load(open(low_cat,'rb')) 
    neg_samples = []
    for seq in X:
        tmat = get_transitions(seq)
        rsum = np.expand_dims(np.sum(tmat, axis=1)+1e-15, axis=1)
        tmat = tmat/rsum
        neg_samples.append(tmat.flatten())
    neg_samples = np.vstack(neg_samples)
    neg_labels = np.zeros((neg_samples.shape[0],))
 
    samples = np.vstack([pos_samples, neg_samples])
    labels = np.vstack([np.expand_dims(pos_labels,axis=1), np.expand_dims(neg_labels, axis=1)]).squeeze()

    # randomly shuffle data
    nsamples = samples.shape[0]
    idx = np.random.permutation(samples.shape[0])
    samples_shuffled = samples[idx,:]
    labels_shuffled = labels[idx]

    accs = []
    # 5 fold cross validation
    for fold in range(5):
        pct_test = int(0.2 * samples_shuffled.shape[0])
        train_idx = np.setdiff1d(np.arange(nsamples), np.arange(fold*pct_test, (fold+1)*pct_test))
        test_idx = np.arange(fold*pct_test, (fold+1)*pct_test)

        train_samples = samples_shuffled[train_idx,:]
        train_labels = labels_shuffled[train_idx]

        test_samples = samples_shuffled[test_idx, :]
        test_labels = labels_shuffled[test_idx]

        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(train_samples, train_labels)

        #import ipdb; ipdb.set_trace()
        preds = clf.predict(test_samples)
        #train_preds = clf.predict(train_samples)
        _acc = acc(test_labels, preds)
        print('Fold: {}, Acc: {}'.format(fold+1, _acc))
        accs.append(_acc)

    return accs
 
def main():

    datasets = ['HI_FC_preexposure_tokens.p', 'HI_FC_postcond_tokens.p', 'HI_FC_postext_tokens.p', 'HI_FC_postret_tokens.p', 'LOW_FC_preexposure_tokens.p', 'LOW_FC_postcond_tokens.p', 'LOW_FC_postext_tokens.p', 'LOW_FC_postret_tokens.p']
        
    titles = ['HI-preexposure', 'HI-postcond', 'HI-postext', 'HI-postret', 'LOW-preexposure', 'LOW-postcond', 'LOW-postext', 'LOW-postret']
    #dataset = 'FC_postcond_tokens.p'
    #title_text = 'Baseline animals (Postconditioning)'
    #dataset = 'FC_tokens.p'
    #title_text = 'Baseline animals (Preconditioning)'
    #dataset = 'FC_postret_tokens.p'
    #title_text = 'Baseline animals (Postret)'

    '''
    ########### Build a HMM
    #import ipdb; ipdb.set_trace()
    remodel = hmm.MultinomialHMM(n_components=3) 
    my_data = [np.expand_dims(x,axis=1) for x in X if x != []]
    lns = [x.shape[0] for x in my_data]
    my_data = np.vstack(my_data).astype(np.uint8)

    remodel.fit(my_data, lns)
    print(remodel.get_stationary_distribution())
    plt.imshow(remodel.transmat_)
    plt.show()
    '''

    '''
    ########### Analyse motif probabilities
    motifs, probs = get_motif_probabilities(norm_trans_mat, trans_mat)
    idx = np.argsort(probs)
    for k in range(idx.shape[0]-1,0,-1):
        mo = motifs[idx[k]]
        my_str = [BEH_LABELS[int(r)] for r in mo]
        my_str = '-'.join(my_str)
        print(my_str, probs[idx[k]])
    '''
    dsets = ['preexposure', 'postcond', 'postret', 'postext']
    mu = []
    var = []
    for dset in dsets:
        accs = fit_model('HI_hcage_FC_{}_tokens.p'.format(dset), 'LOW_hcage_FC_{}_tokens.p'.format(dset))
        mu.append(np.mean(accs))
        var.append(np.std(accs))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(np.arange(len(dsets)), mu, var, label='active periods') #label='all day')

    '''
    dsets = ['preexposure', 'postcond', 'postret', 'postext']
    mu = []
    var = []
    for dset in dsets:
        accs = fit_model('HI_active_FC_{}_tokens.p'.format(dset), 'LOW_active_FC_{}_tokens.p'.format(dset))
        mu.append(np.mean(accs))
        var.append(np.std(accs))
    
    ax.errorbar(np.arange(len(dsets)), mu, var, label='active periods')
    '''

    ax.set_xticks(np.arange(len(dsets)))
    ax.set_xticklabels(dsets, rotation=45)
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Avg. accuracy')
    #ax.set_title('Fear Behavior Clasisifier')
    ax.set_title('hcage fear predictor')
    ax.legend()

    plt.tight_layout()
    #plt.savefig('hcage_fear_pred_classifier.png')
    plt.show()

    #import os; os._exit(0)

    fig, ax = plt.subplots(4,2)
    k = 0
    import ipdb; ipdb.set_trace()
    for dataset in datasets:
        norm_trans_mat = get_trans_mat(dataset)
        np.savetxt('rawTransMats/{}'.format(titles[k]), norm_trans_mat)
        im = ax[k%4, int(k/4)].imshow(norm_trans_mat, cmap="YlGn")
        ax2 = ax[k%4, int(k/4)]

        #cbar = ax2.figure.colorbar(im, ax=ax)
        #cbar.ax.set_ylabel(r'$p(s_{t+1}=j | s_t = i)$', rotation=-90, va="bottom")
        #cbar.outline.set_visible(False)
        
        #ax2.set_xticks(np.arange(len(BEH_LABELS)))
        #ax2.set_xticklabels(BEH_LABELS)

        ax2.set_yticks(np.arange(len(BEH_LABELS)))
        ax2.set_yticklabels(BEH_LABELS)
        for edge, spine in ax2.spines.items():
            spine.set_visible(False)

        '''
        if k%4 == 3:
            ax2.set_xticks(np.arange(len(BEH_LABELS)+1)-.5, minor=True)
            ax2.set_xticklabels(BEH_LABELS)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
        else:
            ax2.set_xticks([])
        '''
        ax2.set_xticks([])

        ax2.set_yticks(np.arange(len(BEH_LABELS)+1)-.5, minor=True)
        ax2.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax2.tick_params(which="minor", bottom=False, left=False)
    
        ax2.set_title(titles[k])
        k += 1

    #plt.title(title_text)
    #fig.tight_layout()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    plt.show()    

if __name__ == '__main__':
    main()
