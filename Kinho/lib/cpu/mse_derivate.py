import numpy as np

def mse_derivate(predicts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """evaluates: error[i] = (2/N) * (predict[i] - target[i])

    Args:
        predicts (np.ndarray): [BATCH][1][N]
        targets (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    ret = []
    
    for predict, target in zip(predicts, targets):
        ret.append(
            2.0 * (predict - target) / predict.shape[1]
        )
    
    return np.array(ret)