from typing import Any, Tuple

import numpy as np


class gradientDescent:
    grad_history: np.ndarray[Any, np.dtype[np.float64]] = np.ndarray([])
    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator, seed: int = 43):
        self.rng = rng or np.random.default_rng(seed=seed)

    def getGradients(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        coefficients: np.ndarray[Any, np.dtype[np.float64]],
        lmbda: float = 0,
        convergence_thresh: float = 0.0001,
        iters: int | None = 300,
        step: float = 0.001,
    ):
        pcoef = coefficients.copy()
        pcoefmask = np.ones_like(pcoef)
        pcoefmask[-1] = 0
        i = 0
        while True:
            grads = -2 * (x.T @ (y - (x @ pcoef))) + (2 * lmbda * pcoef * pcoefmask)
            if np.mean(np.abs(grads)) < convergence_thresh:
                break

            if iters and i >= iters:
                break

            pcoef -= step * grads
            i += 1

        return pcoef

    def getLassoGradientsSubgradient(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        coefficients: np.ndarray[Any, np.dtype[np.float64]],
        lmbda: float = 0,
        convergence_thresh: float = 0.0001,
        iters: int | None = 300,
        step: float = 0.0001,
    ):
        i = 0
        pcoef = coefficients.copy()
        while True:
            pcoefsubgrad = pcoef.copy()
            pcoefsubgrad[-1] = 0
            grads = -2 * (x.T @ (y - (x @ pcoef))) + (lmbda * np.sign(pcoefsubgrad))

            if np.mean(np.abs(grads)) < convergence_thresh:
                return pcoef

            if iters and i >= iters:
                return pcoef

            pcoef -= step * grads
            i += 1

    def getGradientsAutoAdjustLearningRate(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        coefficients: np.ndarray[Any, np.dtype[np.float64]],
        lmbda: int = 0,
        alpha: int = 0,
    ):
        pass
