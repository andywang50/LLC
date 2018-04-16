# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 09:00:38 2017

@author: andy
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle


y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cfm = confusion_matrix(y_true, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#==============================================================================
# plt.figure()
# plt.pcolor(cfm)
# plt.show()
#==============================================================================

cfm = cfm/np.sum(cfm,axis=1)[:,None]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)#plt.cm.ocean
plt.colorbar()
plt.show()

