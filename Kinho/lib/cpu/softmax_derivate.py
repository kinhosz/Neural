import numpy as np

def softmax_derivate(signals: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates: the derivate of softmax

    Softmax's Jacobian isn't diagonal (ds_i/dx_j = s_i*(delta_ij - s_j)), so
    unlike an elementwise activation the upstream gradient for each unit
    depends on every other unit's alpha, not just its own:
    dL/dx_j = s_j * (alpha_j - sum_i(alpha_i * s_i))

    Args:
        signals (np.ndarray): [BATCH][1][N]
        alphas (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """

    ret = []

    for signal, alpha in zip(signals, alphas):
        exp_signal = np.exp(signal)
        s = exp_signal / exp_signal.sum()
        dot = (alpha * s).sum()

        ret.append(s * (alpha - dot))

    return np.array(ret)
