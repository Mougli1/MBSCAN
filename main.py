import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.io import loadmat

from meMatrix import meMatrix

if __name__ == '__main__':
    data_mat = loadmat('data.mat')
    data = data_mat['data']

    F = np.arange(1, data.shape[0] + 1)
    k = 100

    D = cdist(data, data, metric='euclidean')
    sorted_D = np.sort(D, axis=1)
    FScore = sorted_D[:, k]

    j = np.argsort(FScore)
    ndata = data[j, :]

    plt.figure()
    plt.scatter(ndata[:, 0], ndata[:, 1], c=F, s=10, cmap='viridis')
    plt.title('Contour based on distance')
    plt.colorbar()
    plt.show()

    HeightLimit = 8
    NumTree = 100
    e = 1

    MassMatrix, TreeNode, TreeNodeMass, LCA, Forest, Paras = meMatrix(data, NumTree, HeightLimit, e)

    print("Mass-based Dissimilarity (MBD) Matrix:")
    print(MassMatrix)

    np.savetxt('MassMatrix.txt', MassMatrix)

    FScoreM = np.zeros(MassMatrix.shape[0])
    for i in range(MassMatrix.shape[0]):
        D_sorted = np.sort(MassMatrix[i, :])
        FScoreM[i] = D_sorted[k]

    j = np.argsort(FScoreM)
    ndata = data[j, :]

    plt.figure()
    plt.scatter(ndata[:, 0], ndata[:, 1], c=F, s=10, cmap='viridis')
    plt.title('Contour based on m_e')
    plt.colorbar()
    plt.show()

