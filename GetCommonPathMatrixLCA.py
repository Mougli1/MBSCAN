import numpy as np

def GetCommonPathMatrixLCA(DepthLimit):
    NoOfNodes = (2 ** (DepthLimit + 1)) - 1
    Relationship = np.zeros((NoOfNodes, NoOfNodes))
    LCA = np.full((NoOfNodes, NoOfNodes), -1, dtype=int)

    for i in range(NoOfNodes):
        for j in range(i + 1):
            if i == j:
                Relationship[i, j] = int(np.floor(np.log2(i + 1)))
                LCA[i, j] = i
                continue
            height_i = int(np.floor(np.log2(i + 1)))
            height_j = int(np.floor(np.log2(j + 1)))
            root_i = i
            root_j = j
            if height_i != height_j:
                root_i = root_i // (2 ** (height_i - height_j))
            while root_i >= 0:
                if root_i == root_j:
                    Relationship[i, j] = int(np.floor(np.log2(root_i + 1)))
                    Relationship[j, i] = Relationship[i, j]
                    LCA[i, j] = root_i
                    LCA[j, i] = root_i
                    break
                root_i = (root_i - 1) // 2
                root_j = (root_j - 1) // 2
    return Relationship, LCA