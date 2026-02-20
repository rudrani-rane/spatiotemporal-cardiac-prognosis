import torch
import numpy as np

# Basic Regression Metrics

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def mape(preds, targets):
    return torch.mean(torch.abs((targets - preds) / (targets + 1e-8))) * 100

# R2 Score

def r2_score(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    return 1 - (ss_res / (ss_tot + 1e-8))

# Clinical Accuracy

def accuracy_within_tolerance(preds, targets, tol):
    diff = torch.abs(preds - targets)
    correct = (diff <= tol).float()
    return torch.mean(correct).item()

# Aggregate Metrics

def compute_all_metrics(preds, targets):

    preds = preds.view(-1)
    targets = targets.view(-1)

    return {
        "MAE": mae(preds, targets),
        "RMSE": rmse(preds, targets),
        "MAPE": mape(preds, targets).item(),
        "R2": r2_score(preds, targets),

        # Clinical Accuracy
        "Acc@5": accuracy_within_tolerance(preds, targets, 5),
        "Acc@3": accuracy_within_tolerance(preds, targets, 3),
        "Acc@2": accuracy_within_tolerance(preds, targets, 2),
    }