from typing import Any, Dict, Tuple

import numpy as np


def meanSquaredError(
    y: np.array[Tuple[Any], np.dtype[Any]], yhat: np.array[Tuple[Any], np.dtype[Any]]
) -> float:
    return ((y - yhat).T @ (y - yhat)) / y.shape[0]


def precisionRecallAccuracy(
    y: np.array[Tuple[Any], np.dtype[Any]], yhat: np.array[Tuple[Any], np.dtype[Any]]
) -> Dict:
    classes = np.unique(y)
    s = {}
    for c in classes:
        tp = len(y[(y == yhat) & (y == c)])
        tn = len(y[(y == yhat) & (y != c)])
        fp = len(y[(y != c) & (yhat == c)])
        fn = len(y[(y == c) & (yhat != c)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        s[c] = [precision, recall, accuracy]

    return s


def getSSR(
    y: np.array[Tuple[Any], np.dtype[np.float64]],
    y_hat: np.array[Tuple[Any], np.dtype[np.float64]],
) -> float:
    SSR = y - y_hat
    SSR = SSR.T @ SSR
    return SSR


def getRSquared(
    y: np.array[Tuple[Any], np.dtype[np.float64]],
    y_hat: np.array[Tuple[Any], np.dtype[np.float64]],
) -> float:
    mean_SSR = getSSR(y.mean(), y)
    fit_SSR = getSSR(y, y_hat)

    r2 = round((mean_SSR - fit_SSR) / mean_SSR, 2)

    return r2


def getFStatistic(
    y: np.array[Tuple[Any], np.dtype[np.float64]],
    y_hat: np.array[Tuple[Any], np.dtype[np.float64]],
    n: int,
    dof: int,
) -> float:
    r2 = getRSquared(y, y_hat)
    F = (r2 / dof) / ((1 - r2) / (n - dof) - 1)
    return F
