import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from metrics import loss_function


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "Project/climate-learn_set"

single_folder = ["toa_incident_solar_radiation_5.625deg", "2m_temperature_5.625deg", "10m_u_component_of_wind_5.625deg",
                 "10m_v_component_of_wind_5.625deg"]
atmospheric_folder = ["geopotential_5.625deg", "u_component_of_wind_5.625deg", "v_component_of_wind_5.625deg",
                      "temperature_5.625deg", "specific_humidity_5.625deg",
                      "relative_humidity_5.625deg"]

static_variable = "constants_5.625deg"
single_variable = ["toa_incident_solar_radiation_", "2m_temperature_", "10m_u_component_of_wind_",
                   "10m_v_component_of_wind_"]
atmospheric_variable = ["geopotential_", "u_component_of_wind_", "v_component_of_wind_", "temperature_",
                        "specific_humidity_", "relative_humidity_"]

resolution = "_5.625deg"

abbr = ["lsm", "orography", "lat2d", "tisr", "t2m", "u10", "v10", "z", "u", "v", "t", "q", "r"]

levels = [50, 250, 500, 600, 700, 850, 925]
lev_indexes = [0, 4, 7, 8, 9, 10, 11]

low_bound_year_train, max_bound_year_train = 1980, 1983
low_bound_year_val_test, max_bound_year_val_test = 1986, 1987   # First part for validation, second part for test

low_year_train, max_year_train = 0, 2
low_hour_train, max_hour_train = 0, 8759
low_year_val, max_year_val = 0, 0
low_hour_val, max_hour_val = 0, 4379
low_year_test, max_year_test = 0, 0
low_hour_test, max_hour_test = 0, 4379

latitude_coordinates = [-87.1875, -81.5625, -75.9375, -70.3125, -64.6875, -59.0625, -53.4375, -47.8125, -42.1875,
                        -36.5625, -30.9375, -25.3125, -19.6875, -14.0625, -8.4375, -2.8125, 2.8125, 8.4375, 14.0625,
                        19.6875, 25.3125, 30.9375, 36.5625, 42.1875, 47.8125, 53.4375, 59.0625, 64.6875, 70.3125,
                        75.9375, 81.5625, 87.1875]


def return_to_image(x, patch_size, out_channels, img_size):
    p = patch_size
    c = out_channels
    h = img_size[0] // p
    w = img_size[1] // p
    assert h * w == x.shape[1]
    x = torch.reshape(x, shape=(x.shape[0], h, w, p, p, c)).to(device)
    x = torch.einsum("nhwpqc->nchpwq", x).to(device)
    imgs = torch.reshape(x, shape=(x.shape[0], c, (h * p), (w * p))).to(device)
    return imgs


def make_layer(block, in_channels, out_channels, num_blocks, change=0):
    layers = []
    for i in range(num_blocks):
        layers.append(block(in_channels, out_channels))
        if change:
            in_channels = out_channels
    return nn.Sequential(*layers).to(device)


def plot(name, linreg_baseline_rmse, linreg_baseline_acc, resnet_rmse, resnet_acc, unet_rmse, unet_acc, vit_rmse, vit_acc):
    lead_time = [6, 24, 72, 120, 240]
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Plot RMSE on the first subplot
    ax1.plot(lead_time, linreg_baseline_rmse, 'o-', label='Linear Regression RMSE')
    ax1.plot(lead_time, resnet_rmse, 'o-', label='ResNet RMSE')
    ax1.plot(lead_time, unet_rmse, 'o-', label='UNet RMSE')
    ax1.plot(lead_time, vit_rmse, 'o-', label='ViT RMSE')
    ax1.set_xlabel('Leadtime [hours]')
    ax1.set_ylabel('RMSE')
    ax1.set_title(name)
    ax1.legend()
    # Add grid
    ax1.grid(True)
    # Plot ACC on the second subplot
    ax2.plot(lead_time, linreg_baseline_acc, 'o-', label='Linear Regression ACC')
    ax2.plot(lead_time, resnet_acc, 'o-', label='ResNet ACC')
    ax2.plot(lead_time, unet_acc, 'o-', label='UNet ACC')
    ax2.plot(lead_time, vit_acc, 'o-', label='ViT ACC')
    ax2.set_xlabel('Leadtime [hours]')
    ax2.set_ylabel('ACC')
    ax2.set_title(name)
    ax2.legend()
    # Add grid
    ax2.grid(True)
    # Adjust spacing and layout
    plt.tight_layout()
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def create_list(stats):
    l1, l2, l3 = [], [], []
    lists = [l1, l2, l3]
    for i in range(len(lists)):  # Variable
        for j in range(len(stats)):  # Net
            lists[i].append(stats[j][i])
    return lists


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - 1 - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=" ")
    # Print New Line on Complete
    if iteration == total:
        print()


def printProgressAction(action, iteration):
    print(f'\r{action} {iteration}', end=" ")


def EarlyStopping(curr_monitor, old_monitor, count, patience=5, min_delta=0):
    stop = False
    if (curr_monitor - old_monitor) <= min_delta:
        count += 1
        if count > patience:
            stop = True
            return count, stop
    count = 0
    return count, stop


def lr_schedulers(Net_optimizer):
    Net_linearLR = optim.lr_scheduler.LinearLR(Net_optimizer, total_iters=5)
    Net_cos_annLR = optim.lr_scheduler.CosineAnnealingLR(Net_optimizer, T_max=45, eta_min=3.75e-4)
    return Net_linearLR, Net_cos_annLR


def check(Net, data, six_hours_ago, twelve_hours_ago, target, constants, latitude_weights):
    target = target[:, [4, 9, 33], :, :]
    pred = Net(data, six_hours_ago, twelve_hours_ago, target, constants)
    loss = loss_function(pred, target, latitude_weights)
    return loss


class PeriodicPadding2D(nn.Module):
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, inputs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = torch.cat((inputs[:, :, :, -self.pad_width:],
                                   inputs,
                                   inputs[:, :, :, :self.pad_width],), dim=-1, ).to(device)
        # Zero padding in the lat direction
        inputs_padded = nn.functional.pad(inputs_padded, (0, 0, self.pad_width, self.pad_width), ).to(device)
        return inputs_padded


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, lead_time_6h: list, lead_time_24h: list, lead_time_72h: list, lead_time_120h: list,
                 lead_time_240h: list):
        super(CustomDataset, self).__init__()
        # Lead Time 6
        self.data_lt_6 = lead_time_6h[0]
        self.six_hours_ago_lt_6 = lead_time_6h[1]
        self.twelve_hours_ago_lt_6 = lead_time_6h[2]
        self.target_6 = lead_time_6h[3]
        # Lead Time 24
        self.data_lt_24 = lead_time_24h[0]
        self.six_hours_ago_lt_24 = lead_time_24h[1]
        self.twelve_hours_ago_lt_24 = lead_time_24h[2]
        self.target_24 = lead_time_24h[3]
        # Lead Time 72
        self.data_lt_72 = lead_time_72h[0]
        self.six_hours_ago_lt_72 = lead_time_72h[1]
        self.twelve_hours_ago_lt_72 = lead_time_72h[2]
        self.target_72 = lead_time_72h[3]
        # Lead Time 120
        self.data_lt_120 = lead_time_120h[0]
        self.six_hours_ago_lt_120 = lead_time_120h[1]
        self.twelve_hours_ago_lt_120 = lead_time_120h[2]
        self.target_120 = lead_time_120h[3]
        # Lead Time 240
        self.data_lt_240 = lead_time_240h[0]
        self.six_hours_ago_lt_240 = lead_time_240h[1]
        self.twelve_hours_ago_lt_240 = lead_time_240h[2]
        self.target_240 = lead_time_240h[3]

    def __len__(self):
        return len(self.data_lt_240)

    def __getitem__(self, idx):
        l_t_6 = [self.data_lt_6[idx], self.six_hours_ago_lt_6[idx], self.twelve_hours_ago_lt_6[idx],
                 self.target_6[idx]]
        l_t_24 = [self.data_lt_24[idx], self.six_hours_ago_lt_24[idx], self.twelve_hours_ago_lt_24[idx],
                  self.target_24[idx]]
        l_t_72 = [self.data_lt_72[idx], self.six_hours_ago_lt_72[idx], self.twelve_hours_ago_lt_72[idx],
                  self.target_72[idx]]
        l_t_120 = [self.data_lt_120[idx], self.six_hours_ago_lt_120[idx], self.twelve_hours_ago_lt_120[idx],
                   self.target_120[idx]]
        l_t_240 = [self.data_lt_240[idx], self.six_hours_ago_lt_240[idx], self.twelve_hours_ago_lt_240[idx],
                   self.target_240[idx]]
        return l_t_6, l_t_24, l_t_72, l_t_120, l_t_240


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.periodic_zeros_padding = PeriodicPadding2D(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0).to(device)
        self.leaky_relu = nn.LeakyReLU(0.3).to(device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(out_channels).to(device)
        self.shortcut = nn.Identity().to(device)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            ).to(device)

    def forward(self, x):
        residual = self.shortcut(x)
        pad_x = self.periodic_zeros_padding(x)
        x = self.conv1(pad_x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x + residual
        return x
