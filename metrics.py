import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latitude_weighting_function(latitude_coordinates):
    # latitude_coordinates is an array of 'H' elements
    num = np.cos(np.deg2rad(latitude_coordinates))
    den = sum(num) / len(num)
    latitude_weights = num / den
    latitude_weights = torch.from_numpy(latitude_weights).view(1, 1, -1, 1).to(device)
    return latitude_weights.cpu()


def loss_function(prediction, target, latitude_weights):                            # LW_MSE
    error = latitude_weights * np.square(prediction - target)
    result = error.mean([0, 2, 3]).mean(0)

    return result


def LW_RMSE(prediction, target, latitude_weights):
    diff = [x - y for x, y in zip(prediction, target)]
    error = latitude_weights * np.square(diff)      # 32 elements * 1x1x32x64 elements
    channel_rmse = error.mean([2, 3]).sqrt().mean(0)
    result = channel_rmse.mean(1)
    return result.cpu().numpy()


def LW_ACC(prediction, target, latitude_weights, set_climatology):
    climatology = np.asarray(set_climatology).mean(0)
    prediction = prediction - climatology
    target = target - climatology
    channel_acc = []
    for i in range(prediction.shape[1]):
        pred_prime = prediction[:, i] - prediction[:, i].mean()
        target_prime = target[:, i] - target[:, i].mean()
        numer = (latitude_weights * pred_prime * target_prime).sum()
        denom_1 = (latitude_weights * np.square(pred_prime)).sum()
        denom_2 = (latitude_weights * np.square(target_prime)).sum()
        channel_acc.append(numer / np.sqrt(denom_1 * denom_2))
    channel_acc = torch.stack(channel_acc).to(device)
    result = channel_acc
    return result.cpu().numpy()


def compute_eval(prediction, target, latitude_weights, set_climatology):
    rmse = LW_RMSE(prediction, target, latitude_weights)
    acc = LW_ACC(prediction, target, latitude_weights, set_climatology)
    return rmse, acc
