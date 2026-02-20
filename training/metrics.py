import numpy as np
from sklearn.metrics import r2_score


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def R2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def Pearson(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def ConcordanceCC(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def compute_all_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "MAE": MAE(y_true, y_pred),
        "RMSE": RMSE(y_true, y_pred),
        "MAPE": MAPE(y_true, y_pred),
        "R2": R2(y_true, y_pred),
        "Pearson": Pearson(y_true, y_pred),
        "CCC": ConcordanceCC(y_true, y_pred)
    }