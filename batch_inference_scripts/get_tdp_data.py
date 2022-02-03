import os, glob
import h5py
import numpy as np
import tqdm
import copy
import pickle

BASE_PATH = '/gpfs/data/tserre/carney_data/Behavioral_Core_Mice/recordings'

BEH_LABELS = {
        'drink':0, 
        'eat':1, 
        'groom':2, 
        'hang':3, 
        'sniff':4, 
        'rear':5, 
        'rest':6, 
        'walk':7, 
        'eathand':8
        }

FOLDERS = [
'TDP43_Q331K_R1_03_24_16',
'TDP43_Q331K_R2_05_19_16',
'TDP43_Q331K_R2_05_19_16____',
'TDP43_Q331K_R3_07_14_16',
'TDP43_Q331K_R4_09_22_16',
'TDP43_Q331K_R5_11_17_16']

for k in range(len(FOLDERS)):
    full_path = os.path.join(BASE_PATH, FOLDERS[k], 'recordings')

    # get the predictions for this folder
    X = glob.glob(os.path.join(full_path, '*predictions.h5'))
    X = [x for x in X if 'small' not in x]

    # get unique cam IDs
    cams = np.unique([x[x.find('cam')+4:].split('-')[0] for x in X])

    # create the scaffolding
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
   
    for x in tqdm.tqdm(X):
        # find the corresponding camID
        cam_idx = np.where(np.array([cc in x for cc in cams]) == True)[0]
        cur_cam = cams[cam_idx][0]

        f = h5py.File(x, 'r')
        preds = f['predicted_labels'][()]
        pred_data = np.array([BEH_LABELS[p.decode()] for p in preds])
        summary = np.array([np.where(pred_data == x)[0].shape[0] for x in range(9)])
        ntotal = summary.sum()
        summary = summary / ntotal

        parts = x.split('/')[-1].split('_')
        d = parts[3]
        t = parts[4]
        lookup_table[cur_cam][d][t] = summary

    pickle.dump(lookup_table, open('{}.p'.format(FOLDERS[k]),'wb'))
