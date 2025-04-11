import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def get_confusion_matrix(trues: NDArray, preds: NDArray) -> Tuple[int,int,int,int]:
    """ Return (tp, fn, fp, tn)"""
    tp = np.sum(np.logical_and(trues == 1, preds == 1))
    fn = np.sum(np.logical_and(trues == 1, preds == 0))
    fp = np.sum(np.logical_and(trues == 0, preds == 1))
    tn = np.sum(np.logical_and(trues == 0, preds ==0))
    return tp, fn, fp, tn