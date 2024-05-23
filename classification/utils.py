import numpy as np
from typing import Tuple


def med_mad(x: np.ndarray, factor: float=1.4826) -> Tuple[float, float]:
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def rescale_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    med, mad = med_mad(signal)
    signal -= med
    signal /= mad
    return np.clip(signal, -2.5, 2.5)
