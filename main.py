import matplotlib
matplotlib.use('TkAgg')
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import numba as nb
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.io import loadmat

from meMatrix import meMatrix

if __name__ == '__main__':

    data_mat = loadmat('data.mat')
    data = data_mat['data']


    HeightLimit = 8
    NumTree = 100
    e = 1
    np.random.seed(33)  

    MassMatrix, TreeNode, TreeNodeMass, LCA, Forest, Paras = meMatrix(data, NumTree, HeightLimit, e)

    print("Mass-based Dissimilarity (MBD) Matrix:")
    print(MassMatrix)
    sns.heatmap(MassMatrix, annot=False)
    np.savetxt('MassMatrix.txt', MassMatrix)

