import numpy as np
import time
from multiprocessing import Pool
from DisScoreM import DisScoreM
from GenericArrayForest import GenericArrayForest
from GenericArrayEstimation import GenericArrayEstimation
from GetCommonPathMatrixLCA import GetCommonPathMatrixLCA

def compute_dis_score(args):
    A, B, TreeNode, TreeNodeMass, LCA, e = args
    return DisScoreM(A, B, TreeNode, TreeNodeMass, LCA, e)

def meMatrix(data, NumTree, HeightLimit, e):
    print('Construction des iTrees...')
    start_time = time.time()

    Paras = {
        'NumTree': NumTree,
        'HeightLimit': HeightLimit,
        'NumSub': 2 ** HeightLimit,
        'NumDim': data.shape[1],
    }

    if Paras['NumSub'] > data.shape[0]:
        Paras['NumSub'] = data.shape[0]

    Forest = GenericArrayForest(Paras, data)
    EstData = GenericArrayEstimation(data, Forest)
    Relationship, LCA = GetCommonPathMatrixLCA(Paras['HeightLimit'])

    TreeNode = EstData['TreeNode']
    TreeNodeMass = EstData['TreeNodeMass']

    print('iTrees construits.')
    print('Construction de la matrice MBD...')

    n = TreeNode.shape[0]
    Matrix = np.zeros((n, n))

    posr, posc = np.triu_indices(n)

    args_list = [(posr[k], posc[k], TreeNode, TreeNodeMass, LCA, e) for k in range(len(posr))]

    with Pool() as pool:
        p = pool.map(compute_dis_score, args_list)

    Matrix[posr, posc] = p

    Matrix = Matrix + Matrix.T - np.diag(np.diag(Matrix))


    Matrix = Matrix / np.max(Matrix)

    end_time = time.time()
    print(f'Matrice de désimilarité construite en {end_time - start_time:.2f} secondes')

    return Matrix, TreeNode, TreeNodeMass, LCA, Forest, Paras
