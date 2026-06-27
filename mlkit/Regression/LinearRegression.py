from typing import Any, Tuple

import numpy as np
from utils.optimzers import gradientDescent
from utils.stats import getSSR


class normalEquationLinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    rng: np.random.Generator
    lmbda: float = 0

    def __init__(self, seed: int = 43, lmbda: float = 0) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.lmbda = lmbda

    def fit(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fity = y.flatten() if y.shape[-1] == 1 else y

        fitx = np.append(x, padding, axis=-1)
        self.n = fitx.shape[0]

        self.coefficients = (
            np.linalg.inv((fitx.T @ fitx) + (self.lmbda * np.identity(fitx.shape[-1])))
            @ fitx.T
            @ fity
        )

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients


class naiveDescentLinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    rng: np.random.Generator
    lmbda: float = 0

    def __init__(self, seed: int = 43, lmbda: float = 0) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.lmbda = lmbda

    def fit(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        step: float = 0.001,
        convergence_thresh: float = 0.001,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        dim = fitx.shape[-1]

        self.coefficients = self.rng.random((dim), dtype=np.float64)

        fity = y.flatten() if y.shape[-1] == 1 else y
        self.coefficients[-1] = fity.mean()

        preds = self.__predict(fitx)
        last_sum_of_squared_residuals = getSSR(fity, preds)
        global_best_residuals = last_sum_of_squared_residuals
        direction = 1

        for n in range(25):
            for i in range(len(self.coefficients)):
                endurance = 0
                while True:
                    if endurance >= 256:
                        break

                    self.coefficients[i] += direction * step
                    preds = self.__predict(fitx)
                    sum_of_squared_residuals = getSSR(fity, preds) + (
                        self.lmbda * (self.coefficients[i]) ** 2
                    )
                    if sum_of_squared_residuals >= last_sum_of_squared_residuals:
                        direction *= -1
                        endurance += 1
                    elif sum_of_squared_residuals < global_best_residuals:
                        global_best_residuals = sum_of_squared_residuals
                        endurance = 0

                    if (
                        abs(sum_of_squared_residuals - last_sum_of_squared_residuals)
                        <= convergence_thresh
                    ):
                        endurance += 1

                    last_sum_of_squared_residuals = sum_of_squared_residuals

        preds = self.__predict(fitx)
        self.fit_SSR = getSSR(fity, preds)

    def __predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        return x @ self.coefficients

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients


class gradientDescentLinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    rng: np.random.Generator
    lmbda: float = 0

    def __init__(self, seed: int = 43, lmbda: float = 0) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.lmbda = lmbda

    def fit(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        step: float = 0.001,
        convergence_thresh: float = 0.001,
        iters: int | None = None,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        dim = fitx.shape[-1]
        self.coefficients = self.rng.random((dim), dtype=np.float64)

        fity = y.flatten() if y.shape[-1] == 1 else y
        self.coefficients[-1] = fity.mean()

        gd = gradientDescent(self.rng)

        self.coefficients = gd.getGradients(
            fitx, fity, self.coefficients, self.lmbda, convergence_thresh, iters, step
        )

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients


class lassoGradientDescentSubGradientLinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    rng: np.random.Generator
    lmbda: float = 0

    def __init__(self, seed: int = 43, lmbda: float = 0) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.lmbda = lmbda

    def fit(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        step: float = 0.001,
        convergence_thresh: float = 0.001,
        iters: int | None = None,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        self.coefficients = self.rng.random((fitx.shape[1]), dtype=np.float64)

        fity = y.flatten() if y.shape[-1] == 1 else y
        self.coefficients[-1] = fity.mean()
        gd = gradientDescent(self.rng)

        self.coefficients = gd.getLassoGradientsSubgradient(
            fitx, fity, self.coefficients, self.lmbda, convergence_thresh, iters, step
        )

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients


class stochasticGradientDescentLinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    rng: np.random.Generator

    def __init__(self, num_params: int = 1, seed: int = 43) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def fit(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        step: float = 0.001,
        convergence_thresh: float = 0.001,
        iters: int | None = None,
        lmbda: int = 0,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        self.coefficients = self.rng.random((fitx.shape[1]), dtype=np.float64)

        fity = y.flatten() if y.shape[-1] == 1 else y
        self.coefficients[-1] = fity.mean()

        i = 0

        while True:
            grads = getGradients(fitx, fity, self.coefficients, lmbda)

            if np.mean(np.abs(grads)) < convergence_thresh:
                break

            if iters and i >= iters:
                break

            self.coefficients -= step * grads
            i += 1

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients


class lassoStochasticGradientDescentLinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    rng: np.random.Generator

    def __init__(self, num_params: int = 1, seed: int = 43) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def fit(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        step: float = 0.001,
        convergence_thresh: float = 0.001,
        iters: int | None = None,
        lmbda: int = 0,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        self.coefficients = self.rng.random((fitx.shape[1]), dtype=np.float64)

        fity = y.flatten() if y.shape[-1] == 1 else y
        self.coefficients[-1] = fity.mean()
        pass

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients
