import torch

def nrmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    norm_factor = torch.max(y_true) - torch.min(y_true)
    nrmse_value = rmse / norm_factor
    return nrmse_value


def r2_score(y_true, y_pred):
    mean_y_true = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y_true) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
