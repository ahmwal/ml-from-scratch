import numpy as np


def zscoreStandardize(
    x: np.array[Tuple[Any], np.dtype[Any]],
) -> np.array[Tuple[Any], np.dtype[Any]]:
    return (x - x.mean(axis=0)) / x.std(axis=0)


def minMaxNormalize(
    x: np.array[Tuple[Any], np.dtype[Any]],
    a: float | None = None,
    b: float | None = None,
) -> np.array[Tuple[Any], np.dtype[Any]]:
    if a and b:
        return a + ((x - x.min(axis=0)) * (b - a) / (x.max(axis=0) - x.min(axis=0)))

    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
