import numpy as np

def GenericArrayTree(CurtIndex, TreeData, Paras):
    Tree = {
        'MaxNumberOfNodes': Paras['NumberOfTreeNodes'],
        'SplitAttribute': np.zeros(Paras['NumberOfTreeNodes'], dtype=int),
        'SplitPoint': np.zeros(Paras['NumberOfTreeNodes']),
        'TerminationMatrix': np.zeros(Paras['NumSub'], dtype=int),
        'MaxMass': 0,
        'Nodes': np.zeros(Paras['NumberOfTreeNodes'], dtype=int),
        'DataDistribution': [None] * Paras['NumberOfTreeNodes'],
    }

    Tree['Nodes'][0] = Paras['NumSub']
    Tree['DataDistribution'][0] = np.arange(Paras['NumSub'])
    for node in range(Tree['MaxNumberOfNodes']):
        height = int(np.floor(np.log2(node + 1)))
        if Tree['Nodes'][node] <= 1 or height == Paras['HeightLimit']:
            if Tree['Nodes'][node] > 0:
                if height == Paras['HeightLimit']:
                    index = Tree['Nodes'][node] - 1
                    if index < len(Tree['TerminationMatrix']):
                        Tree['TerminationMatrix'][index] += 1
                else:
                    Tree['TerminationMatrix'][0] += 1
                if Tree['MaxMass'] < Tree['Nodes'][node]:
                    Tree['MaxMass'] = Tree['Nodes'][node]
                indices = Tree['DataDistribution'][node]
                if indices is not None:
                    Tree['DataDistribution'][node] = CurtIndex[indices]
            continue
        attempts = 0
        while attempts < 10:
            Tree['SplitAttribute'][node] = np.random.randint(Paras['NumDim'])
            indices = Tree['DataDistribution'][node]
            if indices is None or len(indices) == 0:
                break
            CurtData = TreeData[indices, Tree['SplitAttribute'][node]]
            data_diff = np.max(CurtData) - np.min(CurtData)
            attempts += 1
            if data_diff >= 1e-16:
                Tree['SplitPoint'][node] = np.min(CurtData) + data_diff * np.random.rand()
                break
        else:
            continue
        LeftChild = 2 * node + 1
        RightChild = LeftChild + 1
        if LeftChild >= Tree['MaxNumberOfNodes']:
            continue
        indices = Tree['DataDistribution'][node]
        CurtData = TreeData[indices, Tree['SplitAttribute'][node]]
        left_mask = CurtData < Tree['SplitPoint'][node]
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        Tree['DataDistribution'][LeftChild] = left_indices
        Tree['Nodes'][LeftChild] = len(left_indices)
        Tree['DataDistribution'][RightChild] = right_indices
        Tree['Nodes'][RightChild] = len(right_indices)
        Tree['Nodes'][node] = 0
        Tree['DataDistribution'][node] = None

    return Tree
