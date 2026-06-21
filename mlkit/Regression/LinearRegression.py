from typing import Any, Tuple

import numpy as np


class LinearRegression:
    intercept: float = 0.0
    coefficients: np.array[Tuple[Any], np.dtype[np.float64]] = None
    fit_SSR: float | None = None
    mean_SSR: float | None = None
    n: int = 0
    rng: np.random.Generator

    def __init__(self, num_params: int = 1, seed: int = 43) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.coefficients = self.rng.random((num_params + 1), dtype=np.float64)
        # self.coefficients = np.zeros((num_params + 1,), dtype=np.float64)

    def __getSSR(
        self,
        y: Any,
        yhat: Any,
    ) -> float:
        SSR = y - yhat
        SSR = SSR.T @ SSR
        return SSR

    def fitNaiveStep(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        step: float = 0.001,
        convergence_thresh: float = 0.001,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        self.n = fitx.shape[0]

        fity = y.flatten() if y.shape[-1] == 1 else y
        preds = self.__predict(fitx)

        self.mean_SSR = self.__getSSR(fity.mean(), fity)
        self.coefficients[-1] = fity.mean()

        last_sum_of_squared_residuals = self.__getSSR(fity, preds)
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
                    sum_of_squared_residuals = self.__getSSR(fity, preds)
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
        self.fit_SSR = self.__getSSR(fity, preds)

    def fitClosed(
        self,
        x: np.ndarray[Any, np.dtype[np.float64]],
        y: np.ndarray[Any, np.dtype[np.float64]],
        lmbda: float = 0,
    ) -> None:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fity = y.flatten() if y.shape[-1] == 1 else y
        self.mean_SSR = self.__getSSR(fity.mean(), fity)

        fitx = np.append(x, padding, axis=-1)
        self.n = fitx.shape[0]

        self.coefficients = (
            np.linalg.inv((fitx.T @ fitx) + (lmbda * np.identity(fitx.shape[-1])))
            @ fitx.T
            @ fity
        )
        preds = self.__predict(fitx)
        self.fit_SSR = self.__getSSR(fity, preds)

    def fitGradStep(
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
        self.n = fitx.shape[0]

        fity = y.flatten() if y.shape[-1] == 1 else y
        self.mean_SSR = self.__getSSR(fity.mean(), fity)
        self.coefficients[-1] = fity.mean()

        i = 0

        while True:
            pcoef = self.coefficients.copy()
            pcoef[-1] = 0
            grads = -(2 / fitx.shape[0]) * (
                fitx.T @ (fity - (fitx @ self.coefficients))
            ) + (2 * lmbda * pcoef)

            if np.mean(np.abs(grads)) < convergence_thresh:
                break

            if iters and i >= iters:
                break

            self.coefficients -= step * grads
            i += 1

        preds = self.__predict(fitx)
        self.fit_SSR = self.__getSSR(fity, preds)

    def __predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        return x @ self.coefficients

    def predict(self, x: np.ndarray[Any, np.dtype[np.float64]]) -> np.array:
        padding = np.full(x.shape[:-1] + (1,), 1)
        fitx = np.append(x, padding, axis=-1)
        return fitx @ self.coefficients

    def getRSquared(self) -> dict | None:
        if not (self.mean_SSR and self.fit_SSR):
            raise Exception("Must fit model first before calculating R2!")

        r2 = round((self.mean_SSR - self.fit_SSR) / self.mean_SSR, 2)
        F = (r2 / len(self.coefficients)) / (
            (1 - r2) / (self.n - len(self.coefficients) - 1)
        )

        return {
            "meanSSR": float(self.mean_SSR),
            "fitSSR": float(self.fit_SSR),
            "F": float(F),
            "r2": float(r2),
        }
