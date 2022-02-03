import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score as acc
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import tqdm
from sklearn.manifold import TSNE

def alignKeys(lookup_table, groupHI, groupLOW):
    align_keys = []
    for key in lookup_table:
        found_start = False
        for day in lookup_table[key]:
            for hour in lookup_table[key][day]:
                if type(lookup_table[key][day][hour]) != type(None):
                    print(key, day, hour)
                    if key in groupHI:
                        label = 0
                    elif key in groupLOW:
                        label = 1
                    else:
                        label = 2
                    # align_keys.append((key, day, hour, label))
                    print("Key:", key, " Day:", day, " Hour:", hour, " Label:", label)
                    align_keys.append((key, day, '10h', label))

                    found_start = True
                    break
            if found_start:
                break
    print("Align_Keys: ", align_keys)
    return align_keys

def makeClassificationMatrix(lookup_table, aligned_keys, drop_feature=None):
    N_ANIMALS = 24
    N_HOURS = 50

    if type(drop_feature) == type(None):
        N_BEH = 9
    else:
        N_BEH = 9 - drop_feature.shape[0]

    data_matrix = np.zeros((N_ANIMALS, N_HOURS, N_BEH))
    label_matrix = np.zeros((N_ANIMALS, N_HOURS))

    animal_id = -1
    for idx, tup in enumerate(aligned_keys):
        animal = tup[0]
        start_day = tup[1]
        start_time = tup[2]
        animal_label = tup[3]
        if animal_label == 2:
            continue

        animal_id += 1
        hr_count = 0
        cur_day, cur_time = start_day, start_time
        while hr_count < 48:
            hr = int(cur_time[:-1]) + hr_count
            day = int(cur_day[:-1])

            kd = '%02dD'%(day + np.floor(hr / 25))
            kh = '%02dh'%((hr - 1)% 24 + 1)

            if type(lookup_table[animal][kd][kh]) != type(None):
                if type(drop_feature) == type(None):
                    data_matrix[animal_id, hr_count, :] = lookup_table[animal][kd][kh]
                else:
                    data_matrix[animal_id, hr_count, :] = np.delete(lookup_table[animal][kd][kh], drop_feature)
            label_matrix[animal_id, hr_count] = animal_label

            hr_count += 1
    return data_matrix, label_matrix

def getClassifierAcc(mydata2, mylabs2):
    idx = np.where(np.sum(mydata2, 1) != 0)[0]
    if len(idx) == 0:
       return (-1, -1)

    mydata = mydata2[idx]
    mylabs = mylabs2[idx]

    n_folds = 4
    n_datapoints = mydata.shape[0]
    n_pf = int(n_datapoints/n_folds)

    r_idx = np.random.permutation(n_datapoints)
    val = []
    for fold in range(n_folds):
        test_idx = r_idx[fold*n_pf:(fold+1)*n_pf]
        train_idx = np.setdiff1d(np.arange(n_datapoints), test_idx)

        dat = mydata[train_idx]
        lab = mylabs[train_idx]

        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3, penalty='l1'))
    
        if False:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            tdat = pca.fit_transform(dat)
            ttestdat = pca.transform(mydata[test_idx])
            clf.fit(tdat, lab)
            preds = clf.predict(ttestdat)
        
        #import ipdb; ipdb.set_trace()
        clf.fit(mydata[train_idx], mylabs[train_idx])
        preds = clf.predict(mydata[test_idx])

        val.append(acc(mylabs[test_idx], preds))

    mu, std = np.mean(val), np.std(val)
    return (mu, std)

def runClassification(lookup_table, aligned_keys, drop_feature, make_plot=False):
    data_matrix, label_matrix = makeClassificationMatrix(lookup_table, aligned_keys, drop_feature=drop_feature)

    if make_plot:
        from sklearn.decomposition import PCA
        from mpl_toolkits.mplot3d import Axes3D

        DM = np.reshape(data_matrix, (24*50,-1))
        non_zero = np.where(DM.sum(1) != 0)[0]
        
        pca = PCA(n_components=2)
        pca.fit(DM[non_zero])
        fig = plt.figure()
        
        for k in range(24):
            ax = fig.add_subplot(5,5,k+1) #, projection='3d')
            proj = pca.transform(DM[k*50:(k+1)*50])

            n_s = 24
            #ax.scatter(proj[:24, 0], proj[:24, 1], proj[:24, 2], color=np.array([plt.get_cmap('jet')(x*4) for x in range(24)]))
            #ax.plot(proj[:24, 0], proj[:24, 1], proj[:24, 2], c='k', linewidth=1, alpha=0.25)
            
            ax.scatter(proj[:24, 0], proj[:24, 1], s=8, color=np.array([plt.get_cmap('jet')(x*4) for x in range(24)]))
            ax.plot(proj[:24, 0], proj[:24, 1], c='k', linewidth=1, alpha=0.25)
     
            ax.set_title('%1d'%label_matrix[k][0])
            ax.axis('off')

        plt.show()

    final_mu = []
    for n_run in tqdm.tqdm(range(128)):
        grandMu, grandStd = [], []
        for hr in range(50):
            mu, std = getClassifierAcc(data_matrix[:, hr, :], label_matrix[:, hr])
            if mu == -1 and std == -1:
                continue
            grandMu.append(mu)
            grandStd.append(std)
        final_mu.append(np.array(grandMu))

    acc = np.stack(final_mu)
    mu = np.mean(acc, 0)
    std = np.std(acc, 0)

    return mu, std

if __name__ == '__main__':

    # experimental conditions
    dataset = 'preexposure' #one of ['preexposure', 'postcond', 'postret', 'postext']
    dtype = 'extretav' #one of ['extt1', 'extretav']

    BEH_LABELS = ['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand']

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


    #import ipdb; ipdb.set_trace()
    lookup_table = pickle.load(open(dataset+'.p', 'rb'))
    aligned_keys = alignKeys(lookup_table, groupHI, groupLOW)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mu, std = runClassification(lookup_table, aligned_keys, drop_feature=None)
    ax.plot(np.arange(mu.shape[0]), mu, '-o', alpha=0.75)
    ax.fill_between(np.arange(mu.shape[0]), mu-std, mu+std, alpha=0.25)

    '''
    for k in range(4,9):
        dr_idx = np.setdiff1d(np.arange(9), np.array([k]))
        mu, std = runClassification(lookup_table, aligned_keys, drop_feature=dr_idx)
        ax.plot(np.arange(mu.shape[0]), mu, '-o', alpha=0.75, label='keep {}'.format(BEH_LABELS[k]))
        ax.fill_between(np.arange(mu.shape[0]), mu-std, mu+std, alpha=0.25)
    '''

    ax.legend()        
    ax.plot([0, mu.shape[0]],[0.5, 0.5], 'r--', alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim([0., 1.])

    ax.set_xlabel('Time from onset [in hrs]')
    plt.show()
