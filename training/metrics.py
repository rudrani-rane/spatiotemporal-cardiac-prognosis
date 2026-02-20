import torch
import numpy as np


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def r2_score(pred, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    return (1 - ss_res / ss_tot).item()


def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / (target + 1e-8))).item() * 100


def pearson_corr(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    return np.corrcoef(pred, target)[0, 1]


def concordance_cc(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    mean_pred = np.mean(pred)
    mean_target = np.mean(target)

    var_pred = np.var(pred)
    var_target = np.var(target)

    cov = np.mean((pred - mean_pred) * (target - mean_target))

    return (2 * cov) / (var_pred + var_target + (mean_pred - mean_target) ** 2)