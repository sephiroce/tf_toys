# -*- coding: utf-8 -*-
"""
Created on Jul 15, 2018

@author: Tomer Nahshon
@url: https://medium.com/@tomernahshon/spectral-clustering-from-scratch-38c68968eae0
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
random_state = 21
#X_mn, y_mn = make_moons(150, noise=.07, random_state=random_state)
#X_mn, y_mn = make_circles(150, noise=.07, random_state=random_state)
X_mn, y_mn = make_circles(n_samples=400, factor=.3, noise=0.025)
cmap = 'viridis'
dot_size=50
#fig, ax = plt.subplots(figsize=(9,7))
#ax.set_title('Data with ground truth labels - linear separation not
# possible', fontsize=18, fontweight='demi')
#ax.scatter(X_mn[:, 0], X_mn[:, 1],c=y_mn,s=dot_size, cmap=cmap)
#fig.show()
A = radius_neighbors_graph(X_mn,0.4,mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
# A = kneighbors_graph(X_mn, 2, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False)
A = A.toarray()
print(A.shape)

"""
fig, ax = plt.subplots(figsize=(9,7))
ax.set_title('5 first datapoints', fontsize=18, fontweight='demi')
ax.set_xlim(-1, 2)
ax.set_ylim(-1,1)
ax.scatter(X_mn[:5, 0], X_mn[:5, 1],s=dot_size, cmap=cmap)
for i in range(5):
  ax.annotate(i, (X_mn[i,0],X_mn[i,1]))
fig.show()
"""

L = csgraph.laplacian(A, normed=False)
print(L[:5,:5])

_, eigvec = np.linalg.eig(L)
#np.where(eigval == np.partition(eigval, 1)[1])# the second smallest eigenvalue

#print(eigvec)
"""
If the graph is fully connected, lambda 2 (eigenvalue number 2) is greater than 
0 and represents tha algebraeic connectivity of the our graph. The greater the
eigenvalue (lambda 2) the more connected the graph is.
"""
# second eigenvalue
y_spec = eigvec[:,1].copy()
#print(y_spec)
y_spec[y_spec < 0] = 0
y_spec[y_spec > 0] = 1

print(type(y_spec))
print(y_mn.shape)
print(y_spec.shape)

fig, ax = plt.subplots(figsize=(9,7))
ax.set_title('Data after spectral clustering from scatch', fontsize=18,
             fontweight='demi')
ax.scatter(X_mn[:, 0], X_mn[:, 1],c=y_spec ,s=dot_size, cmap=cmap)
fig.show()

model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
 assign_labels='kmeans')
labelsS = model.fit_predict(X_mn)
fig, ax = plt.subplots(figsize=(9,7))
ax.set_title('kernal transform to higher dimension\nlinear separation is '
             'possible', fontsize=18, fontweight='demi')
plt.scatter(X_mn[:, 0], X_mn[:, 1], c=labelsS, s=dot_size, cmap=cmap)
fig.show()