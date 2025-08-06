import numpy as np

def bce_derivate(predicts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """BCE - Binary Cross Entropy Derivate

        BCE = (1.0 / BATCH) * SUM(1<=i<=BATCH)[ (P[i] - Y[i]) / (P[i] * (1.0 - P[i])) ]

        where:
            Y[i] = target[i]
            P[i] = predict[i]

    Args:
        predicts (np.ndarray): [BATCH][1][N]
        targets (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    BATCH = predicts.shape[0]

    res = (predicts - targets) / (predicts * (1.0 - predicts))
    res = res / BATCH

    return res
