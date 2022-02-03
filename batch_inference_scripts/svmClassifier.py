import numpy as np
import glob
from hmmlearn import hmm
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

BEH_LABELS = ['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand']

'''grep all the LSTM model predictions
'''
def getFiles(base_path):
    files = []
    for path in Path(base_path).rglob('*.csv'):
        files.append(path)
    return files

def parseFile(file_path):
    with open(file_path,'r') as f:
        pred_data = f.readlines()
    if len(pred_data) > 110000:
        return [], []

    pred_data = np.array([int(x.split(',')[-1]) for x in pred_data])
    #pred_data = scipy.ndimage.filters.maximum_filter1d(pred_data, 15)

    global_summary = np.array([np.where(pred_data == x)[0].shape[0] for x in range(9)])
    ntotal = global_summary.sum()
    global_summary = global_summary / ntotal
    '''
    if global_summary[6] < 0.6:
        frame_counter = -79
        vid = cv2.VideoCapture(os.path.join('/media/data_cifs/nih/files_to_send', '/'.join(file_path.split('/')[-2:]).replace('.csv','.mp4')))
        while vid.isOpened():
            ret, frame = vid.read()
            if frame_counter >= 0:
                image = cv2.putText(frame, BEH_LABELS[pred_data[frame_counter]], (20,20), cv2.FONT_HERSHEY_SIMPLEX , 1., (0, 0, 255),2, cv2.LINE_AA, False)
                cv2.imshow('predictions', image) 
                cv2.waitKey(1)
            frame_counter += 1
            if frame_counter >= pred_data.shape[0]:
                break
        vid.close()
    '''
    return global_summary, pred_data

def makeSummaryPlot(summary_mat, file_names, exp_name, cam_name, bin_width=10):
    exp_mat = np.dstack([summary_mat] * bin_width).reshape(summary_mat.shape[0],-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(exp_mat)
    ax.set_title(exp_name, fontsize=8)
    ax.set_xticks(np.arange(0.5, 9, 1)*bin_width)
    ax.set_yticks(np.arange(len(file_names)))
    ax.set_xticklabels(['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand'], fontsize=8)
    ax.set_yticklabels([x.split('/')[-1].split('.')[0] for x in file_names], fontsize=5)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')    
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())

    plt.show(block=False)
    plt.savefig(exp_name+'_cam_'+cam_name+'.png')

def run(pred_datafiles, exp_name):
    cur_exps = [x for x in pred_datafiles if exp_name in x.parts]
    cams = np.unique([x.name.split('_')[-1].split('-')[0] for x in cur_exps])
    cam_idx = 0

    cur_exp_cam = [x for x in cur_exps if cams[cam_idx] in x.name]
    sorted_videos = np.sort([x.name for x in cur_exp_cam])
    base_path = cur_exp_cam[0].parent
    summaries, file_names, predictions = [], [], []

    for data_file in tqdm.tqdm(sorted_videos):
        summary, preds = parseFile(os.path.join(base_path,data_file))
        if len(summary) != 0:
            summaries.append(summary)
            file_names.append(data_file)
            predictions.append(preds)

    summary_mat = np.array(summaries)
    #pred_mat = np.vstack(predictions)
    #makeSummaryPlot(summary_mat, file_names, exp_name, cams[cam_idx])
    return summary_mat, predictions

def collapse(X, time_bin):
    ans = []
    for i in range(0, X.shape[0], time_bin):
        ans.append([scipy.stats.mode(X[i:i+time_bin]).mode[0]])
    return ans

def processExperiment(pred_datafiles, experiment):
    summaries = []
    predictions = []
    for ename in experiment:
        summary_mat, pred_mat = run(pred_datafiles, ename)
        summaries.append(summary_mat)
        predictions.extend(pred_mat)

    time_bin = 30
    collapsed_pred = [collapse(x, time_bin) for x in predictions]    
    lengths = [len(x) for x in collapsed_pred]
    X = np.vstack(collapsed_pred)

    remodel = hmm.GaussianHMM(n_components=5, covariance_type='full', n_iter=100) 
    remodel.monitor_.verbose = True
    remodel.fit(X, lengths)   
    '''
    states = []
    for x in collapsed_pred:
        states.append(remodel.predict(x))
    plt.imshow(remodel.transmat_); plt.show()
    '''
    return remodel.transmat_

def alignPredictions(pred_datafiles, exp, base_path):
    cur_exps = []
    for exp_name in exp:
        cur_exps.extend([x for x in pred_datafiles if exp_name in x.parts])

    suffix = str.lower(exp[0].split('_')[0].split('-')[-1])
    cams = np.unique([x.name.split('_')[-1].split('-')[0]+'_'+suffix for x in cur_exps])
    lookup_table = dict.fromkeys(cams)

    # add the days
    schedule = {}
    for d in range(31):
        schedule.update({'%02dD'%(d+1): None})

    for key in schedule.keys():
        sch_days = {}
        for h in range(24):
            sch_days.update({'%02dh'%(h+1): None})
        schedule.update({key: copy.deepcopy(sch_days)})

    for key in lookup_table.keys():
        lookup_table.update({key: copy.deepcopy(schedule)})

    for cam in cams:
        camID = cam[:-2] # removing the suffix
        cur_exp_cam = [x for x in cur_exps if camID in x.name]
        sorted_videos = np.sort(['/'.join(x.parts[-2:]) for x in cur_exp_cam])

        for data_file in tqdm.tqdm(sorted_videos):
            summary, preds = parseFile(os.path.join(base_path,data_file))
            
            if len(summary) != 0:
                parts = data_file.split('/')[-1].split('_')
                d = parts[3]
                t = parts[4]
                lookup_table[cam][d][t] = summary
    return lookup_table

def makeChart(lookup_table, exp_title):
    # add the days
    schedule = {}
    for d in range(31):
        schedule.update({'%02dD'%(d+1): None})
    # add the hours
    for key in schedule.keys():
        sch_days = {}
        for h in range(24):
            sch_days.update({'%02dh'%(h+1): None})
        schedule[key] = copy.deepcopy(sch_days)

    is_start = False
    for kday in schedule.keys():
        for khour in schedule[kday].keys():
            n_cam = 0
            for kcam in lookup_table.keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    n_cam += 1
                    schedule[kday][khour] = summary_stat if (schedule[kday][khour] is None) else schedule[kday][khour] + summary_stat
            schedule[kday][khour] = schedule[kday][khour]/n_cam if n_cam != 0 else None

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap = matplotlib.cm.get_cmap('Set1')

    preds = []
    labels = []
    indices = []
    idx = 0
    for kday in schedule.keys():
        for khour in schedule[kday].keys():
            z = schedule[kday][khour]
            if z is None:
                idx = idx+1 if idx !=0 else 0
                continue
            preds.append(z)
            labels.append('{}_{}'.format(kday,khour))

    all_preds = np.vstack(preds)
    X = scipy.ndimage.filters.gaussian_filter1d(all_preds, 3., 0)
    for k in range(X.shape[1]):
        ax.plot(X[:,k], c=cmap(k))

    ax.legend(BEH_LABELS)
    ax.set_xticks(np.arange(len(preds)))
    ax.set_xticklabels(labels, rotation=45, fontsize=6)
    ax.set_ylabel('% of time spent in this behavior')
    ax.set_title(exp_title)

    #plt.get_current_fig_manager().window.state('zoomed')
    mng = plt.get_current_fig_manager() 
    mng.resize(*mng.window.maxsize())
    plt.show(block=False)
    plt.savefig(os.path.join('results', '{}_timeline.png'.format(exp_title)))
    plt.close()

def makePieChart(lookup_table, exp_title, writer, fields):
    # add the days
    schedule = {}
    for d in range(31):
        schedule.update({'%02dD'%(d+1): None})
    # add the hours
    for key in schedule.keys():
        sch_days = {}
        for h in range(24):
            sch_days.update({'%02dh'%(h+1): None})
        schedule[key] = copy.deepcopy(sch_days)

    global_summary = dict.fromkeys(lookup_table.keys()) 
    global_count = dict.fromkeys(lookup_table.keys()) 
    for k in global_summary.keys():
        global_summary[k] = np.zeros((9,))
        global_count[k] = 0.

    is_start = False
    for kday in schedule.keys():
        for khour in schedule[kday].keys():
            n_cam = 0
            for kcam in lookup_table.keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    global_summary[kcam] += summary_stat
                    global_count[kcam] += 1

    for kcam in global_summary.keys():
        global_summary[kcam] /= global_count[kcam]
        row = {}
        for k in range(11):
            if k == 0:
                row.update({fields[0]: exp_title})
            elif k == 1:
                row.update({fields[1]: kcam})
            else:
                row.update({fields[k]: global_summary[kcam][k-2]})
        writer.writerow(row) 

    n_cams = len(global_summary.keys())
    n_cols = 6
    n_rows = np.int(np.ceil(n_cams/n_cols))

    fig = plt.figure()
    cmap = matplotlib.cm.get_cmap('Set1')

    plt_idx = 1
    for kcam in global_summary.keys():
        ax = fig.add_subplot(n_rows, n_cols, plt_idx)
        ax.set_prop_cycle('color', [cmap(k) for k in range(len(BEH_LABELS))])
        wedges, texts, autotexts = ax.pie(global_summary[kcam], autopct='%1.1f%%',
                                  textprops=dict(color="w"))
        ax.set_title(kcam)
        plt_idx += 1
        plt.setp(autotexts, size=5, weight='bold')
    
    ax.legend(wedges, BEH_LABELS, title='Action', loc='center left', bbox_to_anchor=(1,0,0.5,1))
    mng = plt.get_current_fig_manager() 
    mng.resize(*mng.window.maxsize())
    plt.show(block=False)
    plt.savefig(os.path.join('results', '{}_per_animal.png'.format(exp_title)))
    plt.close()

def makeDayAvgChart(lookup_table, save_name, exp_title, writer, fields):
    avg_day = {}
    avg_cnts = {}
    for h in range(24):
        #avg_day.update({'%02dh'%(h): np.zeros((9,))})
        avg_day.update({'%02dh'%h : []})
        avg_cnts.update({'%02dh'%(h): 0.})

    ### curate the list of cams available on that recording time
    for kcam in lookup_table.keys():
        for kday in lookup_table[kcam].keys():
            for khour in lookup_table[kcam][kday].keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    avg_day[khour].append(summary_stat)
                    avg_cnts[khour] += 1

    ### compute final day averages and variances
    preds, stddev = [], []
    hr_avg = {}
    for khour in avg_day.keys():
        if avg_cnts[khour] == 0.:
            hr_avg.update({khour : '-' })
            continue
        X = np.vstack(avg_day[khour]) # = avg_day[khour] / avg_cnts[khour]
        mu = np.mean(X, axis=0)
        hr_avg.update({khour: mu})
        sig = np.std(X, axis=0) 
        preds.append(mu)
        stddev.append(sig)

    ### visualize this
    preds = np.vstack(preds)
    stddev = np.vstack(stddev)

    X = scipy.ndimage.filters.gaussian_filter1d(preds, 1., 0)
    err = scipy.ndimage.filters.gaussian_filter1d(stddev, 1., 0)

    cmap = matplotlib.cm.get_cmap('Set1')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(X.shape[1]):
        ax.fill_between(np.arange(X.shape[0]), X[:,k]-err[:,k], X[:,k]+err[:, k], alpha=0.1, color=cmap(k))
        ax.plot(X[:, k], c=cmap(k), linewidth=4, label=BEH_LABELS[k])

    ax.legend()
    ax.set_xticks(np.arange(X.shape[0]))
    ax.set_xticklabels([x for x in avg_day.keys()], rotation=45, fontsize=10, fontweight='bold')
    ax.set_ylabel('% of time spent in this behavior',fontsize=16,  fontweight='bold')
    ax.set_title(exp_title)

    mng = plt.get_current_fig_manager() 
    mng.resize(*mng.window.maxsize())
    plt.show(block=False)
    plt.savefig(os.path.join('results', '{}_per_day_timeline.png'.format(save_name)))
    plt.close()

    for khour in hr_avg.keys():
        row = {}
        for k in range(11):
            if k == 0:
                row.update({fields[0]: exp_title})
            elif k == 1:
                row.update({fields[1]: khour})
            else:
                if type('-') == type(hr_avg[khour]):
                    row.update({fields[k]: '-'})
                else: 
                    row.update({fields[k]: hr_avg[khour][k-2]})
        writer.writerow(row)


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
            six_hr_summaries.update({kcam: summary.flatten()})

    return six_hr_summaries

def get4HrSummary(lookup_table):

    six_hr_summaries = {}
    ### curate the list of cams available on that recording time
    for kcam in lookup_table.keys():
        six_hour_summary = np.zeros((8,9))
        six_hour_counts = np.zeros((8,))
        for kday in lookup_table[kcam].keys():
            for khour in lookup_table[kcam][kday].keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    hr = int(int(khour[:-1])/3)
                    six_hour_summary[hr,:] += summary_stat
                    six_hour_counts[hr] += 1
        summary = six_hour_summary/np.expand_dims(six_hour_counts, -1)
        six_hr_summaries.update({kcam: summary.flatten()})

    return six_hr_summaries


def run(
    base_path = '/media/data_cifs/projects/prj_nih/prj_andrew_holmes/inference/batch_inference/model_predictions/',
    n_bootstrap = 4098,
    mode = 'class', 
    datasets = None,
    shuffle = False,
    dtype = 'extretav',
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
        summary = get6HrSummary(lookup_table, drop_feature=drop_feature)
        #summary = get4HrSummary(lookup_table)

        tmp_summary = copy.deepcopy(summary)
        for key in tmp_summary.keys():
            if not(key in fvalues.keys()):
                tmp = summary.pop(key, None)
                control_summary.update({key: tmp})

        n_datapoints = len(summary.keys())
        r_idx = np.random.permutation(n_datapoints)
        #r_idx = np.arange(n_datapoints)

        keys = list(summary.keys())
        control_keys = list(control_summary.keys())
        control_data = np.vstack([control_summary[x] for x in control_keys])
        control_labels = np.array([1 for x in control_keys])

        n_folds = 4 #8
        n_pf = int(n_datapoints/n_folds)
    
        all_accs = []
        all_coefs = []
        all_controls = []

        for brun in tqdm.tqdm(range(n_bootstrap)):
            accs = []
            c_accs = []
            for fold in range(n_folds):
                test_idx = r_idx[fold*n_pf:(fold+1)*n_pf]
                train_idx = np.setdiff1d(np.arange(n_datapoints), test_idx)

                train_keys = [keys[x] for x in train_idx]
                test_keys = [keys[x] for x in test_idx]

                train_data = np.vstack([summary[x] for x in train_keys])
                test_data = np.vstack([summary[x] for x in test_keys])
                
                if mode == 'regression':
                    train_labels = np.array([fvalues[x] for x in train_keys])
                    test_labels = np.array([fvalues[x] for x in test_keys])
                    
                    clf = make_pipeline(StandardScaler(), linear_model.LinearRegression())
                    acc_fn = mean_squared_error
                else:
                    train_labels = np.array([2*int(x in groupLOW)-1 for x in train_keys])
                    test_labels = np.array([2*int(x in groupLOW)-1 for x in test_keys])
                    if shuffle == True:
                        train_labels = np.random.permutation(train_labels)
                        test_labels = np.random.permutation(test_labels)

                    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3, penalty='l1'))
                    #clf = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', C=1000.))
                    #clf = make_pipeline(StandardScaler(), svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, tol=1e-3))
                    acc_fn = acc

                clf.fit(train_data, train_labels)
                preds = clf.predict(test_data)
                control_preds = clf.predict(control_data)

                all_coefs.append(clf[1].coef_)

                _acc = acc_fn(test_labels, preds)
                control_acc = acc_fn(control_labels, control_preds)

                #accs.append(np.sqrt(_acc))
                accs.append(_acc)
                c_accs.append(control_acc)

            #print('Mean accuracy: {}'.format(np.mean(accs)))
            all_accs.append(np.mean(accs))
            all_controls.append(np.mean(c_accs))

        if plot_importance:
            coefs = np.mean(np.vstack(all_coefs), axis=0)
            coefs = np.reshape(coefs, (-1,9))

            labs = ['12AM -- 6AM', '6AM -- Noon', 'Noon -- 6PM', '6PM -- 12PM']
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for rr in range(coefs.shape[0]):
                ax.bar(np.arange(coefs[rr].shape[0])+ 0.1*rr, coefs[rr], 0.1, label=labs[rr])
                ax.set_xticks(np.arange(9)+0.2)
                ax.set_xticklabels(BEH_LABELS)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel('Behavior')
            plt.ylabel('Weight')
            plt.title('Phase: {}'.format(datasets[0][1]))
            plt.legend()
            plt.savefig('featureweights_phase_{}.png'.format(datasets[0][1]))
            plt.show()

        print('Acc: {}'.format(np.mean(all_accs)))
        print('Std: {}'.format(np.std(all_accs)))
        np.save('{}_dtype_{}_shuffle_{}_v2'.format(datasets[0][1], dtype, shuffle), all_accs)
        #np.save('{}_dtype_{}_shuffle_{}_control_v2'.format(datasets[0][1], dtype, shuffle), all_controls)
    else:
        for dset in datasets:

            # suffix to disambiguate file name?
            filt = None
            if len(dset) > 2:
               filt = dset[2]

            # get all the prediction files from this category
            if filt != None:
                exp = [x.split('/')[-1] for x in all_folders if (str.lower(dset[0]) in str.lower(x)) and (str.lower(dset[1]) in str.lower(x)) and (str.lower(filt) not in str.lower(x))]
            else:
                exp = [x.split('/')[-1] for x in all_folders if (str.lower(dset[0]) in str.lower(x)) and (str.lower(dset[1]) in str.lower(x))]
            name = dset[0] + '_' + dset[1]

            table = alignPredictions(pred_datafiles, exp, base_path)
            lookup_table.update(copy.deepcopy(table))

        pickle.dump(lookup_table, open(datasets[0][1]+'.p', 'wb'))
        os._exit(0)


if __name__ == '__main__':

    # need a list of folder names for a given experiment and phase
    all_datasets = [[('FC-A', 'preexposure', 'Trap'), ('FC-B','preexposure'), ('FC-C', 'preexposure'), ('FC-D', 'preexposure')],
                [('FC-A', 'postcond', 'Trap'), ('FC-B','postcond'), ('FC-C', 'postcond'), ('FC-D', 'postcond')],
                [('FC-A', 'postret', 'Trap'), ('FC-B','postret'), ('FC-C', 'postret'), ('FC-D', 'postret')],
                [('FC-A', 'postext', 'Trap'), ('FC-B','postext'), ('FC-C', 'postext'), ('FC-D', 'postext')]]

    for k in range(len(all_datasets)):
        run(datasets=all_datasets[k], dtype='extretav', shuffle=True)

    #run(datasets=all_datasets[0], dtype='extretav')

    #for k in range(9):
    #    run(datasets=all_datasets[0], drop_feature=k)
 
