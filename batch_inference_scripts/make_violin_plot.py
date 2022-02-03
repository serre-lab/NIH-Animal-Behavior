import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats

experiments = ['preexposure', 'postcond', 'postret', 'postext']
shuffle = ['True', 'False']
base_dir = 'feat_6hr_win'
dtype = 'extretav'

my_dict = {'mean accuracy': [], 'phase': [], 'shuffled': []}

for exp in experiments:
    for sh in shuffle:
        #data = np.load(os.path.join(base_dir, '{}_dtype_{}_shuffle_{}_v2.npy'.format(exp, dtype, sh)))
        data = np.load('{}_dtype_{}_shuffle_{}_v2.npy'.format(exp, dtype, sh))
        my_dict['mean accuracy'].extend(data.tolist())
        my_dict['phase'].extend([exp] * data.shape[0])
        my_dict['shuffled'].extend([sh] * data.shape[0])

    '''
    data = np.load(os.path.join(base_dir, '{}_dtype_{}_shuffle_{}_control.npy'.format(exp, dtype, 'False')))
    my_dict['mean accuracy'].extend(data.tolist())
    my_dict['phase'].extend([exp] * data.shape[0])
    my_dict['shuffled'].extend(['control'] * data.shape[0])
    #my_dict['control'].extend(['True'] * data.shape[0])
    '''

my_df = pd.DataFrame(my_dict)

sns.boxplot(x='phase', y='mean accuracy', data=my_df, hue='shuffled', palette='Set3')

for xval, exp in enumerate(experiments):
    null_v = my_df['mean accuracy'][(my_df['phase'] == exp) & (my_df['shuffled']=='True')]
    actual_v = my_df['mean accuracy'][(my_df['phase'] == exp) & (my_df['shuffled']=='False')]

    nq1, nq2, nq3 = np.quantile(null_v, 0.25), np.quantile(null_v, 0.5), np.quantile(null_v, 0.75)
    niqr = (nq3 - nq1) * 1.5
    noutliers = null_v[null_v < nq1-niqr].tolist() + null_v[null_v > nq3+niqr].tolist() 

    q1, q2, q3 = np.quantile(actual_v, 0.25), np.quantile(actual_v, 0.5), np.quantile(actual_v, 0.75)
    iqr = (q3 - q1) * 1.5
    outliers = actual_v[actual_v < q1-iqr].tolist() + actual_v[actual_v > q3+iqr].tolist() 

    with open('boxstats_{}_{}.txt'.format(exp, dtype), 'w') as f:
        f.write('null_q0: {}, null_q1: {}, null_q2: {}, null_q3: {}, null_q4: {}\n'.format(nq1-niqr, nq1, nq2, nq3, nq3+niqr))
        f.write('null_outliers: {}\n'.format(noutliers))
        f.write('q0: {}, q1: {}, q2: {}, q3: {}, q4: {}\n'.format(q1-iqr, q1, q2, q3, q3+iqr))
        f.write('outliers: {}\n'.format(outliers))

    kstest = stats.ks_2samp(null_v, actual_v, alternative='greater')
    pval = kstest.pvalue

    annot_str = ''
    if pval < 0.001:
        annot_str = '***'
    elif pval < 0.01:
        annot_str = '**'
    elif pval < 0.05:
        annot_str = '*'
    else:
        annot_str = 'ns'

    x1, x2 = xval-0.25, xval+0.25  
    y, h, col = my_df['mean accuracy'].max() + 0.01, 0.01, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, annot_str, ha='center', va='bottom', color=col)

#plt.savefig('{}_predictive_power.png'.format(dtype))
plt.title(dtype)
plt.show()
