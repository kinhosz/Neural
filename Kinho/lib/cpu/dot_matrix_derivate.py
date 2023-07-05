import numpy as np

def dot_matrix_derivate(const_matrix: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates: alpha.dot(const_matrix.transpose())

    Args:
        const_matrix (np.ndarray): [N][M]
        alphas (np.ndarray): [BATCH][1][M]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    
    ret = []
    
    for alpha in alphas:
        ret.append(
            alpha.dot(const_matrix.transpose())
        )
    
    return np.array(ret)
