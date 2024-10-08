import numpy as np
from GenericArrayTree import GenericArrayTree

def GenericArrayForest(Paras, Data):
    Forest = {
        'Trees': [],
        'NumInst': Data.shape[0],
        'DimInst': Data.shape[1],
        'generated_trees': 0,
        'NumberOfTreeNodes': (2 ** (Paras['HeightLimit'] + 1)) - 1,
        'NumTree': Paras['NumTree'],
    }
    Paras['NumberOfTreeNodes'] = Forest['NumberOfTreeNodes']

    np.random.seed()

    for _ in range(Paras['NumTree']):
        Forest['generated_trees'] += 1
        CurtIndex = np.random.choice(Forest['NumInst'], Paras['NumSub'], replace=False)
        TreeData = Data[CurtIndex, :]
        Tree = GenericArrayTree(CurtIndex, TreeData, Paras)
        Forest['Trees'].append(Tree)

    return Forest