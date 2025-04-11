import numpy as np
from sklearn.metrics import roc_auc_score

def safe_auc(y_true, y_pred):
    raw_score = roc_auc_score(y_true, y_pred, average=None)
    if np.isnan(raw_score):
        return 0.5
    return raw_score