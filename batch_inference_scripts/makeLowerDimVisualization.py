import numpy as np
import glob
from pathlib import Path
import csv, tqdm, os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import cv2, scipy
import pickle
import copy
import csv
import matplotlib

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score as acc
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import tqdm
from sklearn.manifold import TSNE

BEH_LABELS = ['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand']

def getFiles(base_path):
    files = []
    for path in Path(base_path).rglob('*.csv'):
        files.append(path)
    return files


def get6HrSummary(lookup_table, drop_feature=None):

    six_hr_summaries = {}
    ### curate the list of cams available on that recording time
    for kcam in lookup_table.keys():
        six_hour_summary = np.zeros((4,9))
        six_hour_counts = np.zeros((4,))
        for kday in lookup_table[kcam].keys():
            for khour in lookup_table[kcam][kday].keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    hr = int(int(khour[:-1])/6)
                    six_hour_summary[hr,:] += summary_stat
                    six_hour_counts[hr] += 1
        summary = six_hour_summary/np.expand_dims(six_hour_counts, -1)

        if drop_feature != None:
            dsummary = np.delete(summary, drop_feature, axis=1)
            six_hr_summaries.update({kcam: dsummary.flatten()})
        else:
            #six_hr_summaries.update({kcam: summary.flatten()})
            six_hr_summaries.update({kcam: summary})

    return six_hr_summaries

def getHRSummary(lookup_table, drop_feature=None):

    hr_summaries = {}
    ### curate the list of cams available on that recording time
    for kcam in lookup_table.keys():
        hour_summary = np.zeros((24,9))
        hour_counts = np.zeros((24,))
        for kday in lookup_table[kcam].keys():
            for khour in lookup_table[kcam][kday].keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    hr = int(khour[:-1])
                    hour_summary[hr,:] += summary_stat
                    hour_counts[hr] += 1
        summary = hour_summary/np.expand_dims(hour_counts, -1)

        if drop_feature != None:
            dsummary = np.delete(summary, drop_feature, axis=1)
            six_hr_summaries.update({kcam: dsummary.flatten()})
        else:
            hr_summaries.update({kcam: summary})

    return hr_summaries

def getTTSplit(summary, pca, groupHI, groupLOW):
    n_folds = 4
    n_datapoints = 24 #len(summary.keys())
    n_pf = int(n_datapoints/n_folds)

    r_idx = np.random.permutation(n_datapoints)
    keys = list(summary.keys())
    all_data, all_labels = [], []

    for key in keys:
        dat = pca.transform(summary[key]).flatten()
        if key in groupHI:
            all_data.append(dat)
            all_labels.append(0)
        elif key in groupLOW:
            all_data.append(dat)
            all_labels.append(1)
    aData = np.stack(all_data)
    aLabels = np.array(all_labels)
    val = []

    for fold in range(n_folds):
        test_idx = r_idx[fold*n_pf:(fold+1)*n_pf]
        train_idx = np.setdiff1d(np.arange(n_datapoints), test_idx)

        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3, penalty='l1'))
        clf.fit(aData[train_idx], aLabels[train_idx])
        preds = clf.predict(aData[test_idx])
        val.append(acc(aLabels[test_idx], preds))

    return np.mean(val)

def visTSNE(summary, groupHI, groupLOW):
    X, lab = [], []
    for key in summary.keys():
        if key in groupHI:
            X.append(summary[key])
            lab.append(np.zeros((summary[key].shape[0],)))
        if key in groupLOW:
            X.append(summary[key])
            lab.append(np.zeros((summary[key].shape[0],))+1)
    X = np.vstack(X)
    labs = np.concatenate(lab)

    tsne = TSNE(n_components=2)
    X_emb = tsne.fit_transform(X)
    plt.scatter(X_emb[:,0], X_emb[:,1], c=labs)
    plt.show()

    import ipdb; ipdb.set_trace()

def run(
    base_path = '/media/data_cifs/projects/prj_nih/prj_andrew_holmes/inference/batch_inference/model_predictions/',
    n_bootstrap = 1000,
    mode = 'class', 
    datasets = None,
    shuffle = False,
    dtype = 'extt1',
    plot_importance = False,
    drop_feature = None
    ):
    
    pred_datafiles = getFiles(base_path)

    if dtype == 'extretav':
        # fear after extinction   
        groupHI = ['17202338_a', '17202345_b', '17202338_b', '17202346_b', '17202341_b', '17202341_c', '17202346_c', '17202342_d', '6394837_d','17202341_d', '17202339_d', '6394836_d', '6394841_d']
        groupLOW = ['17202339_a', '17202342_a', '17202346_a', '17202341_a', '17202345_a', '17202339_b','17202342_b','17202338_c', '17202345_c', '17202339_c', '17202342_c']
    elif dtype == 'extt1':
        # fear during conditioning
        groupHI = ['17202339_a', '17202342_a', '17202346_a', '17202341_a','17202345_b', '17202338_b', '17202346_c', '17202345_c', '17202339_c', '17202342_c', '17202342_d', '6394837_d']
        groupLOW = ['17202338_a', '17202345_a', '17202346_b', '17202346_b', '17202341_b', '17202339_b','17202342_b','17202341_c', '17202338_c', '17202341_d', '17202339_d', '6394836_d', '6394841_d']
    else:
        raise NotImplementedError

    fvalues = {
    '17202338_a': 37, '17202345_b': 47, '17202338_b': 30, '17202346_b': 30, '17202341_b': 33, '17202341_c': 30, '17202346_c': 53, '17202342_d': 33, '6394837_d': 40,'17202341_d': 47, '17202339_d': 40, '6394836_d': 50, '6394841_d': 40, '17202339_a': 23, '17202342_a': 27, '17202346_a': 27, '17202341_a': 13, '17202345_a': 20, '17202339_b': 20,'17202342_b': 20,'17202338_c':20, '17202345_c':17, '17202339_c':27, '17202342_c':23
    }

    '''
    fvalues = {
    '17202338_a': 30, '17202345_b': 53, '17202338_b': 77, '17202346_b': 43, '17202341_b': 43, '17202341_c': 40, '17202346_c': 63, '17202342_d': 87, '6394837_d': 67,'17202341_d': 40, '17202339_d': 37, '6394836_d': 20, '6394841_d': 47, '17202339_a': 73, '17202342_a': 77, '17202346_a': 67, '17202341_a': 53, '17202345_a': 47, '17202339_b': 37,'17202342_b': 30,'17202338_c':47, '17202345_c':50, '17202339_c':53, '17202342_c':50
    }
    '''

    all_folders = glob.glob(os.path.join(base_path,'*'))
    lookup_table = {}

    if os.path.exists(datasets[0][1]+'.p'):
        lookup_table = pickle.load(open(datasets[0][1]+'.p','rb'))
        control_summary = {}
        
        summary = getHRSummary(lookup_table)
        #summary = get6HrSummary(lookup_table)

        visTSNE(summary, groupHI, groupLOW)

        X, labs = [], []
        for key in summary.keys():
            X.append(summary[key])
        allF = np.vstack(X)

        pca = PCA(n_components=3)
        pca.fit(allF)

        '''
        vs = []
        for k in range(1000):
            vs.append(getTTSplit(summary, pca, groupHI, groupLOW))
        print(np.mean(vs))
        import ipdb; ipdb.set_trace()
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111) #, projection='3d')

        agg_hi = np.zeros((24,9))
        agg_hi_count = 0
        agg_low = np.zeros((24,9))
        agg_low_count = 0

        for key in summary.keys():
            if key in groupHI:
                agg_hi += summary[key]
                agg_hi_count += 1
            if key in groupLOW:
                agg_low += summary[key]
                agg_low_count += 1

        agg_hi /= agg_hi_count
        agg_low /= agg_low_count

        import ipdb; ipdb.set_trace()

        tH = pca.transform(agg_hi)
        tL = pca.transform(agg_low)

        ax.plot(tH[:, 0], tH[:, 1], 'ro-', alpha=0.5)
        ax.plot(tL[:, 0], tL[:, 1], 'go-', alpha=0.5)
        plt.show()

        for key in summary.keys():
            tX = pca.transform(summary[key])
            myc = matplotlib.cm.Greys(np.linspace(0, 1, 24))
            if key in groupHI:
                myc = matplotlib.cm.Blues(np.linspace(0, 1, 24))
            elif key in groupLOW:
                myc = matplotlib.cm.Greens(np.linspace(0, 1, 24))

            #ax.scatter(tX[:,0], tX[:, 1], tX[:, 2], c=myc)
            ax.plot(tX[:, 0], tX[:, 1], tX[:, 2], c= myc[-1], alpha=0.5)
            #ax.plot(tX[:,0], tX[:, 1], c=myc[-1], alpha=0.5)

        plt.show()

if __name__ == '__main__':

    # need a list of folder names for a given experiment and phase
    all_datasets = [[('FC-A', 'preexposure', 'Trap'), ('FC-B','preexposure'), ('FC-C', 'preexposure'), ('FC-D', 'preexposure')],
                [('FC-A', 'postcond', 'Trap'), ('FC-B','postcond'), ('FC-C', 'postcond'), ('FC-D', 'postcond')],
                [('FC-A', 'postret', 'Trap'), ('FC-B','postret'), ('FC-C', 'postret'), ('FC-D', 'postret')],
                [('FC-A', 'postext', 'Trap'), ('FC-B','postext'), ('FC-C', 'postext'), ('FC-D', 'postext')]]

    run(datasets=all_datasets[1], dtype='extt1', shuffle=False)

    #for k in range(len(all_datasets)):
    #    run(datasets=all_datasets[k])

    #for k in range(9):
    #    run(datasets=all_datasets[0], drop_feature=k)
