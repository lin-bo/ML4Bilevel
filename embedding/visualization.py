import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def multi2two_dim(feature, method='pca'):
    if method == 'svd':
        centered = feature - np.mean(feature, axis=0)
        covariance = 1.0 / feature.shape[0] * centered.T.dot(centered)
        U, S, V = np.linalg.svd(covariance)
        coord = centered.dot(U[:, 0:2])
    elif method == 'pca':
        coord = PCA(random_state=0).fit_transform(feature)[:, :2]
    return coord


def two_dim_visual(feature, coreset, neighbors):
    # check dimension
    n, d = feature.shape
    if d != 2:
        raise ValueError('The given feature is {} dimensional, should be 2 dimensional'.format(d))
    if len(neighbors) > 0 and len(neighbors[0]) != 1:
        raise ValueError('Only allow one neighbor for each point')
    # exrtact points
    coreset_pnt = feature[coreset]
    flag = np.ones(n, dtype=bool)
    flag[coreset] = False
    other_pnt = feature[flag]
    # visualize points
    plt.scatter(coreset_pnt.T[0], coreset_pnt.T[1], color='r', label='coreset point', alpha=0.7)
    plt.scatter(other_pnt.T[0], other_pnt.T[1], color='b', label='other point', alpha=0.3)
    # visualize connections
    if len(neighbors) > 0:
        for idx, neighbor in enumerate(neighbors):
            pnts = feature[[idx, neighbor[0]]]
            plt.plot(pnts.T[0], pnts.T[1], color='b', alpha=0.3)
    plt.legend()
    plt.show()
