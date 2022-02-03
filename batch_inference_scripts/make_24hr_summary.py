import numpy as np
import glob
from pathlib import Path
import csv, tqdm, os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2, scipy
import pickle
import copy
import csv
import matplotlib
import pandas as pd

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

    cams = np.unique([x.name.split('_')[-1].split('-')[0] for x in cur_exps])
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
        cur_exp_cam = [x for x in cur_exps if cam in x.name]
        sorted_videos = np.sort(['/'.join(x.parts[-2:]) for x in cur_exp_cam])

        for data_file in tqdm.tqdm(sorted_videos):
            summary, preds = parseFile(os.path.join(base_path,data_file))
            
            if len(summary) != 0:
                parts = data_file.split('/')[-1].split('_')
                d = parts[3]
                t = parts[4]
                lookup_table[cam][d][t] = summary
    return lookup_table

def dryRunAlignPredictions(pred_datafiles, exp, base_path):
    cur_exps = []
    for exp_name in exp:
        cur_exps.extend([x for x in pred_datafiles if exp_name in x.parts])

    cams = np.unique([x.name.split('_')[-1].split('-')[0] for x in cur_exps])
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
        cur_exp_cam = [x for x in cur_exps if cam in x.name]
        sorted_videos = np.sort(['/'.join(x.parts[-2:]) for x in cur_exp_cam])

        print('{} | {} | {}'.format(exp, cam, len(sorted_videos)))
        '''
        for data_file in tqdm.tqdm(sorted_videos):
            summary, preds = parseFile(os.path.join(base_path,data_file))
            
            if len(summary) != 0:
                parts = data_file.split('/')[-1].split('_')
                d = parts[3]
                t = parts[4]
                lookup_table[cam][d][t] = summary
        '''

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
    plt.savefig(os.path.join('fallon_results', '{}_per_day_timeline.png'.format(save_name)))
    plt.close()

    for khour in hr_avg.keys():
        row = {}
        for k in range(10):
            if k == 0:
                row.update({fields[0]: khour})
            else:
                if type('-') == type(hr_avg[khour]):
                    row.update({fields[k]: '-'})
                else: 
                    row.update({fields[k]: hr_avg[khour][k-1]*3600.})
        writer.writerow(row)

def makePerDayAvgChartPerAnimalPATRICKROEE(lookup_table, dataset, writer, df):

    # each camera ID corresponds to a different animal. so let's compute the 24 HR
    # average for each of these 
    for kcam in lookup_table.keys():
        for kday in lookup_table[kcam].keys():

            avg_day = {}
            avg_cnts = {}
            for h in range(24):
                avg_day.update({'%02dh'%h : []})
                avg_cnts.update({'%02dh'%(h): 0.})

            should_record = False
            for khour in lookup_table[kcam][kday].keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    should_record = True
                    avg_day[khour].append(summary_stat)
                    avg_cnts[khour] += 1

            if not should_record:
                continue

            ### compute final day averages and variances
            #preds, stddev = [], []
            hr_avg = {}
            for khour in avg_day.keys():
                if avg_cnts[khour] == 0.:
                    hr_avg.update({khour : '-' })
                    continue
                X = np.vstack(avg_day[khour]) # = avg_day[khour] / avg_cnts[khour]
                mu = np.mean(X, axis=0)
                hr_avg.update({khour: mu})

                #sig = np.std(X, axis=0) 
                #preds.append(mu)
                #stddev.append(sig)

            if dataset not in np.unique(df['Folder']):
                continue

            print('Processing {}'.format(dataset))
            rec_dates = np.unique(df[df['Folder'] == dataset]['dates of recording'])[0]
            start_month = int(rec_dates.split('-')[0].split('/')[0])
            start_day = int(rec_dates.split('-')[0].split('/')[-1])

            end_month = int(rec_dates.split('-')[-1].split('/')[0])
            end_day = int(rec_dates.split('-')[-1].split('/')[1])
            year = int(rec_dates.split('-')[-1].split('/')[-1])

            cur_day = int(kday[:-1])

            if start_month == end_month:
                day_count = cur_day - start_day + 1
                my_date = '{}/{}/{}'.format(start_month, cur_day, year)
            else:
                dpm = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

                if cur_day < start_day:
                    day_count = (dpm[start_month-1] - start_day + 1) + cur_day
                    my_date = '{}/{}/{}'.format(end_month, cur_day, year)
                else:
                    day_count = cur_day - start_day + 1
                    my_date = '{}/{}/{}'.format(start_month, cur_day, year)

            gender = 'male'
            age = np.unique(df[df['Folder'] == dataset ]['age'])[0]

            df2 = df[df['Folder'] == dataset]
            df3 = df2[df2['Cam_ID'] == int(kcam)]
            if len(df3) != 1:
                mouse_id = kcam 
                genotype = '-'
                continue
            else:
                mouse_id = df3['Mouse_ID'].to_numpy()[0]
                genotype = df3['genotype '].to_numpy()[0]


            ### write out the CSV file
            for khour in hr_avg.keys():
                row = {}
                row.update({'Mouse ID': mouse_id, 'Genotype': genotype, 'Gender': gender, 'Age Group': age, 'Day of Recording': day_count,  'Hour (0-23)': int(khour[:-1]), 'Date of recording session': my_date})
                # check if whole row is dummy
                should_write = False
                for k in range(9):
                    if type('-') != type(hr_avg[khour]):
                        should_write = True
                        break

                if not should_write:
                    continue

                for k in range(9):
                    if type('-') == type(hr_avg[khour]):
                        row.update({BEH_LABELS[k]: '-'})
                    else: 
                        row.update({BEH_LABELS[k]: hr_avg[khour][k]*3600.})
                writer.writerow(row)

def makePerDayAvgChartPerAnimal(lookup_table, dataset):

    # each camera ID corresponds to a different animal. so let's compute the 24 HR
    # average for each of these 
    for kcam in lookup_table.keys():
        for kday in lookup_table[kcam].keys():

            avg_day = {}
            avg_cnts = {}
            for h in range(24):
                avg_day.update({'%02dh'%h : []})
                avg_cnts.update({'%02dh'%(h): 0.})

            should_record = False
            for khour in lookup_table[kcam][kday].keys():
                summary_stat = lookup_table[kcam][kday][khour]
                if summary_stat is None:
                    continue
                else:
                    should_record = True
                    avg_day[khour].append(summary_stat)
                    avg_cnts[khour] += 1

            if should_record:
                csvfile = open('fallon_results_v2/{}_cam_{}_day_{}_stats.csv'.format(dataset, kcam, kday), 'w+')
                fieldnames = ['hour'] + BEH_LABELS 
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            else:
                continue

            ### compute final day averages and variances
            #preds, stddev = [], []
            hr_avg = {}
            for khour in avg_day.keys():
                if avg_cnts[khour] == 0.:
                    hr_avg.update({khour : '-' })
                    continue
                X = np.vstack(avg_day[khour]) # = avg_day[khour] / avg_cnts[khour]
                mu = np.mean(X, axis=0)
                hr_avg.update({khour: mu})

                #sig = np.std(X, axis=0) 
                #preds.append(mu)
                #stddev.append(sig)

            ### write out the CSV file
            for khour in hr_avg.keys():
                row = {}
                for k in range(10):
                    if k == 0:
                        row.update({fieldnames[0]: khour})
                    else:
                        if type('-') == type(hr_avg[khour]):
                            row.update({fieldnames[k]: '-'})
                        else: 
                            row.update({fieldnames[k]: hr_avg[khour][k-1]*3600.})
                writer.writerow(row)
            csvfile.close()

def makeDayAvgChartPerAnimal(lookup_table, dataset):

    # each camera ID corresponds to a different animal. so let's compute the 24 HR
    # average for each of these 
    for kcam in lookup_table.keys():

        csvfile = open('fallon_results/{}_cam_{}_stats.csv'.format(dataset, kcam),'w+')
        fieldnames = ['hour'] + BEH_LABELS 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        avg_day = {}
        avg_cnts = {}
        for h in range(24):
            avg_day.update({'%02dh'%h : []})
            avg_cnts.update({'%02dh'%(h): 0.})

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
        import ipdb; ipdb.set_trace()
        ### write out the CSV file
        for khour in hr_avg.keys():
            row = {}
            for k in range(10):
                if k == 0:
                    row.update({fieldnames[0]: khour})
                else:
                    if type('-') == type(hr_avg[khour]):
                        row.update({fieldnames[k]: '-'})
                    else: 
                        row.update({fieldnames[k]: hr_avg[khour][k-1]*3600.})
            writer.writerow(row)
        csvfile.close()

if __name__ == '__main__':
    #base_path = '/gpfs/data/tserre/carney_data/Behavioral_Core_Mice/recordings/fallon_G85R_prediction_analysis/fallon_lstm_results'
    base_path = '/gpfs/data/tserre/data/lgovinda/fallon_G85R_prediction_analysis/fallon_lstm_results'
    datasets = os.listdir(base_path)

    pred_datafiles = getFiles(base_path)
    all_folders = glob.glob(os.path.join(base_path,'*'))

    df = pd.read_csv('fallon_g85r_metadata.csv')

    csvfile = open('AllFallonDataOct2021.csv', 'w+')
    fieldnames = ['Mouse ID', 'Genotype', 'Gender', 'Age Group', 'Day of Recording',  'Hour (0-23)', 'Date of recording session'] + BEH_LABELS 
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for dataset in datasets:
        lookup_table = alignPredictions(pred_datafiles, [dataset], base_path)
        #dryRunAlignPredictions(pred_datafiles, [dataset], base_path)        

        #makeDayAvgChart(lookup_table, dataset, 'Experiment: {}'.format(dataset), writer, fieldnames)
        #makeDayAvgChartPerAnimal(lookup_table, dataset)

        ## this was used to generate the per-animal stats
        #makePerDayAvgChartPerAnimal(lookup_table, dataset)

        ## this was used to generate the collated CSV
        makePerDayAvgChartPerAnimalPATRICKROEE(lookup_table, dataset, writer, df)

    csvfile.close()
