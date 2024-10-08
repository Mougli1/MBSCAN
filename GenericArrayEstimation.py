import numpy as np

def GenericArrayEstimation(Data, Forest):
    EstData = {
        'NumInst': Data.shape[0],
        'NumTree': Forest['NumTree'],
        'NumberOfTreeNodes': Forest['NumberOfTreeNodes'],
        'TreeNode': np.zeros((Data.shape[0], Forest['NumTree']), dtype=int),
        'TreeNodeMass': np.zeros((Forest['NumTree'], Forest['NumberOfTreeNodes'])),
    }

    for k in range(Forest['NumTree']):
        TreeNodeIndx = [None] * Forest['Trees'][k]['MaxNumberOfNodes']
        TreeNodeIndx[0] = np.arange(EstData['NumInst'])
        TreeNodeMass = np.zeros(Forest['NumberOfTreeNodes'])
        for node in range(Forest['Trees'][k]['MaxNumberOfNodes']):
            indices = TreeNodeIndx[node]
            if indices is None or len(indices) == 0:
                continue
            TreeNodeMass[node] += len(indices)
            EstData['TreeNodeMass'][k, node] = TreeNodeMass[node]
            if Forest['Trees'][k]['Nodes'][node] > 0:
                EstData['TreeNode'][indices, k] = node
            else:
                LeftChild = 2 * node + 1
                RightChild = LeftChild + 1
                if LeftChild >= Forest['Trees'][k]['MaxNumberOfNodes']:
                    continue
                CurtData = Data[indices, Forest['Trees'][k]['SplitAttribute'][node]]
                left_mask = CurtData < Forest['Trees'][k]['SplitPoint'][node]
                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]
                TreeNodeIndx[LeftChild] = left_indices
                TreeNodeIndx[RightChild] = right_indices
    return EstData

