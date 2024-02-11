import argparse
import torch
from utils import latitude_coordinates, define_times, create_list, plot
from .forecasting import Forecasting
from data_processing import *
from metrics import latitude_weighting_function, compute_eval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


low_year_test, max_year_test = 0, 0
low_hour_test, max_hour_test = 0, 4379


def evaluate(n_epochs=10):
    train_set, validation_set, test_set = define_sets()
    latitude_weights = latitude_weighting_function(latitude_coordinates)
    forecasters = Forecasting(train_set, validation_set)
    forecasters.load()

    pred_ResNet_6, pred_ResNet_24, pred_ResNet_72, pred_ResNet_120, pred_ResNet_240 = ([], ) * 5
    targ_ResNet_6, targ_ResNet_24, targ_ResNet_72, targ_ResNet_120, targ_ResNet_240 = ([], ) * 5
    pred_UNet_6, pred_UNet_24, pred_UNet_72, pred_UNet_120, pred_UNet_240 = ([], ) * 5
    targ_UNet_6, targ_UNet_24, targ_UNet_72, targ_UNet_120, targ_UNet_240 = ([], ) * 5
    pred_ViT_6, pred_ViT_24, pred_ViT_72, pred_ViT_120, pred_ViT_240 = ([], ) * 5
    targ_ViT_6, targ_ViT_24, targ_ViT_72, targ_ViT_120, targ_ViT_240 = ([], ) * 5

    print("Start Evaluation")
    for i in range(n_epochs):
        print("Epoch : ", i)
        # time_x = [year, hour, year_6, hour_6, year_12, hour_12, year_targ, hour_targ]
        lead_times = [6, 24, 72, 120, 240]
        time_6, time_24, time_72, time_120, time_240 = define_times(low_year_test, max_year_test, low_hour_test,
                                                                    max_hour_test, lead_times)
        # ResNets
        p, t = forecasters.ResNet_6.Net(test_set, low_year_test, max_year_test, 6, time_6)
        pred_ResNet_6.append(p), targ_ResNet_6.append(t)
        p, t = forecasters.ResNet_24.Net(test_set, low_year_test, max_year_test, 24, time_24)
        pred_ResNet_24.append(p), targ_ResNet_24.append(t)
        p, t = forecasters.ResNet_72.Net(test_set, low_year_test, max_year_test, 72, time_72)
        pred_ResNet_72.append(p), targ_ResNet_72.append(t)
        p, t = forecasters.ResNet_120.Net(test_set, low_year_test, max_year_test, 120, time_120)
        pred_ResNet_120.append(p), targ_ResNet_120.append(t)
        p, t = forecasters.ResNet_240.Net(test_set, low_year_test, max_year_test, 240, time_240)
        pred_ResNet_240.append(p), targ_ResNet_240.append(t)
        # UNets
        p, t = forecasters.UNet_6.Net(test_set, low_year_test, max_year_test, 6, time_6)
        pred_UNet_6.append(p), targ_UNet_6.append(t)
        p, t = forecasters.UNet_24.Net(test_set, low_year_test, max_year_test, 24, time_24)
        pred_UNet_24.append(p), targ_UNet_24.append(t)
        p, t = forecasters.UNet_72.Net(test_set, low_year_test, max_year_test, 72, time_72)
        pred_UNet_72.append(p), targ_UNet_72.append(t)
        p, t = forecasters.UNet_120.Net(test_set, low_year_test, max_year_test, 120, time_120)
        pred_UNet_120.append(p), targ_UNet_120.append(t)
        p, t = forecasters.UNet_240.Net(test_set, low_year_test, max_year_test, 240, time_240)
        pred_UNet_240.append(p), targ_UNet_240.append(t)
        # ViT
        p, t = forecasters.ViT_6.Net(test_set, low_year_test, max_year_test, 6, time_6)
        pred_ViT_6.append(p), targ_ViT_6.append(t)
        p, t = forecasters.ViT_24.Net(test_set, low_year_test, max_year_test, 24, time_24)
        pred_ViT_24.append(p), targ_ViT_24.append(t)
        p, t = forecasters.ViT_72.Net(test_set, low_year_test, max_year_test, 72, time_72)
        pred_ViT_72.append(p), targ_ViT_72.append(t)
        p, t = forecasters.ViT_120.Net(test_set, low_year_test, max_year_test, 120, time_120)
        pred_ViT_120.append(p), targ_ViT_120.append(t)
        p, t = forecasters.ViT_240.Net(test_set, low_year_test, max_year_test, 240, time_240)
        pred_ViT_240.append(p), targ_ViT_240.append(t)
    # ResNets
    rmse_ResNet_6, acc_ResNet_6 = compute_eval(pred_ResNet_6, targ_ResNet_6, latitude_weights, forecasters.ResNet_6.Net.set_climatology)
    rmse_ResNet_24, acc_ResNet_24 = compute_eval(pred_ResNet_24, targ_ResNet_24, latitude_weights, forecasters.ResNet_24.Net.set_climatology)
    rmse_ResNet_72, acc_ResNet_72 = compute_eval(pred_ResNet_72, targ_ResNet_72, latitude_weights, forecasters.ResNet_72.Net.set_climatology)
    rmse_ResNet_120, acc_ResNet_120 = compute_eval(pred_ResNet_120, targ_ResNet_120, latitude_weights, forecasters.ResNet_120.Net.set_climatology)
    rmse_ResNet_240, acc_ResNet_240 = compute_eval(pred_ResNet_240, targ_ResNet_240, latitude_weights, forecasters.ResNet_240.Net.set_climatology)
    # UNets
    rmse_UNet_6, acc_UNet_6 = compute_eval(pred_UNet_6, targ_UNet_6, latitude_weights, forecasters.UNet_6.Net.set_climatology)
    rmse_UNet_24, acc_UNet_24 = compute_eval(pred_UNet_24, targ_UNet_24, latitude_weights, forecasters.UNet_24.Net.set_climatology)
    rmse_UNet_72, acc_UNet_72 = compute_eval(pred_UNet_72, targ_UNet_72, latitude_weights, forecasters.UNet_72.Net.set_climatology)
    rmse_UNet_120, acc_UNet_120 = compute_eval(pred_UNet_120, targ_UNet_120, latitude_weights, forecasters.UNet_120.Net.set_climatology)
    rmse_UNet_240, acc_UNet_240 = compute_eval(pred_UNet_240, targ_UNet_240, latitude_weights, forecasters.UNet_240.Net.set_climatology)
    # ViT
    rmse_ViT_6, acc_ViT_6 = compute_eval(pred_ViT_6, targ_ViT_6, latitude_weights, forecasters.ViT_6.Net.set_climatology)
    rmse_ViT_24, acc_ViT_24 = compute_eval(pred_ViT_24, targ_ViT_24, latitude_weights, forecasters.ViT_24.Net.set_climatology)
    rmse_ViT_72, acc_ViT_72 = compute_eval(pred_ViT_72, targ_ViT_72, latitude_weights, forecasters.ViT_72.Net.set_climatology)
    rmse_ViT_120, acc_ViT_120 = compute_eval(pred_ViT_120, targ_ViT_120, latitude_weights, forecasters.ViT_120.Net.set_climatology)
    rmse_ViT_240, acc_ViT_240 = compute_eval(pred_ViT_240, targ_ViT_240, latitude_weights, forecasters.ViT_240.Net.set_climatology)

    resnet_rmse = create_list([rmse_ResNet_6, rmse_ResNet_24, rmse_ResNet_72, rmse_ResNet_120, rmse_ResNet_240])
    resnet_acc = create_list([acc_ResNet_6, acc_ResNet_24, acc_ResNet_72, acc_ResNet_120, acc_ResNet_240])
    unet_rmse = create_list([rmse_UNet_6, rmse_UNet_24, rmse_UNet_72, rmse_UNet_120, rmse_UNet_240])
    unet_acc = create_list([acc_UNet_6, acc_UNet_24, acc_UNet_72, acc_UNet_120, acc_UNet_240])
    vit_rmse = create_list([rmse_ViT_6, rmse_ViT_24, rmse_ViT_72, rmse_ViT_120, rmse_ViT_240])
    vit_acc = create_list([acc_ViT_6, acc_ViT_24, acc_ViT_72, acc_ViT_120, acc_ViT_240])

    plot('t2m', resnet_rmse[0], resnet_acc[0], unet_rmse[0], unet_acc[0], vit_rmse[0], vit_acc[0])
    plot('Z500', resnet_rmse[1], resnet_acc[1], unet_rmse[1], unet_acc[1], vit_rmse[1], vit_acc[1])
    plot('T850', resnet_rmse[2], resnet_acc[2], unet_rmse[2], unet_acc[2], vit_rmse[2], vit_acc[2])


def train():
    train_set, validation_set, test_set = define_sets()
    forecasters = Forecasting(train_set, validation_set).to(device)
    forecasters.train_forecasters()
    forecasters.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate()


if __name__ == '__main__':
    main()
