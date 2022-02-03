import numpy as np
import matplotlib.pyplot as plt

X1 = np.load('preexposure_regr_ret.npy')
X2 = np.load('postcond_regr_ret.npy')
X3 = np.load('postext_regr_ret.npy')
X4 = np.load('postret_regr_ret.npy')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.boxplot([X1,X2,X3,X4],showmeans=True)
ax.plot([1,2,3,4], [np.mean(X1), np.mean(X2), np.mean(X3), np.mean(X4)])
ax.set_xticklabels(['preexposure', 'postcond', 'postext', 'postret'])
ax.set_ylabel('mean error')
plt.show()
