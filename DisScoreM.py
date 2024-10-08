import numpy as np

def DisScoreM(A, B, TreeNode, TreeNodeMass, LCA, e):
    n = TreeNode.shape[1]
    score = np.zeros(n)
    for i in range(n):
        nodeA = TreeNode[A, i]
        nodeB = TreeNode[B, i]
        lca_node = LCA[nodeA, nodeB]
        mass = TreeNodeMass[i, lca_node]
        score[i] = mass ** e
    SimiScore = (np.mean(score)) ** (1 / e)
    return SimiScore

