import torch
import torch.nn as nn
import numpy.random as rnd
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path = "E:/Universita/2 Magistrale/18 CFU/2 Neural Networks/Project/Implementation/climate-learn"

single_folder = ["toa_incident_solar_radiation_5.625deg", "2m_temperature_5.625deg", "10m_u_component_of_wind_5.625deg", "10m_v_component_of_wind_5.625deg"]
atmospheric_folder = ["geopotential_5.625deg", "u_component_of_wind_5.625deg", "v_component_of_wind_5.625deg", "temperature_5.625deg", "specific_humidity_5.625deg",
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

low_year_train, max_year_train = 1979, 1985                             # Original values from paper: 1979, 2016
low_year_val_test, max_year_val_test = 1985, 1986

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


def extractor(data, year, hour, year_6=None, hour_6=None, year_12=None, hour_12=None, for_pred=True):
    dataset = []
    if for_pred:
        for i in range(len(abbr)):
            # Static Variables (constants)
            if i < 3:  # 3
                dataset.append(data[i][0])
            # Single Variables
            if 2 < i < 7:   # 4
                dataset.append(data[i][year][hour])
                dataset.append(data[i][year_6][hour_6])
                dataset.append(data[i][year_12][hour_12])
            # Atmospheric Variables
            if 6 < i < 13:  # 6
                for j in range(len(levels)):    # 7
                    dataset.append(data[i][year][j][hour])
                    dataset.append(data[i][year_6][j][hour_6])
                    dataset.append(data[i][year_12][j][hour_12])

    else:
        # No Static Variables (constants)
        # 2 Metre Temperatur (Single Variables)
        dataset.append(data[4][year][hour])
        # Geopotential and Temperature (Atmospheric Variables)
        dataset.append(data[7][year][2][hour])
        dataset.append(data[10][year][5][hour])

    return dataset


def compute_years_hours(low_year, max_year, low_hour_test, max_hour_test, lead_time):
    year = rnd.randint(low_year, (max_year + 1))
    if year == max_year:
        hour = rnd.randint(low_hour_test, (max_hour_test - lead_time))
    elif year == low_year:
        hour = rnd.randint(low_hour_test + 12, max_hour_test)
    else:
        hour = rnd.randint(low_hour_test, max_hour_test)

    if year != low_year and (hour - 6) < 0:
        year_6 = year - 1
        hour_6 = max_hour_test + (hour - 6)
        year_12 = year_6
        hour_12 = max_hour_test + (hour - 12)
    elif year != low_year and (hour - 12) < 0:
        year_12 = year - 1
        hour_12 = max_hour_test + (hour - 12)
        if (hour - 6) < 0:
            year_6 = year - 1
            hour_6 = max_hour_test + (hour - 6)
        else:
            year_6 = year
            hour_6 = hour - 6
    else:
        year_6 = year
        year_12 = year
        hour_12 = hour - 12
        hour_6 = hour - 6

    if (hour + lead_time > max_hour_test - 1) and (year != max_year):
        year_targ = year + 1
        hour_targ = (hour + lead_time) - max_hour_test
    else:
        year_targ = year
        hour_targ = hour + lead_time

    return year, hour, year_6, hour_6, year_12, hour_12, year_targ, hour_targ


def define_times(low_year, max_year, low_hour, max_hour, lead_times: list):
    times = []
    for i in range(len(lead_times)):
        times.append(compute_years_hours(low_year, max_year, low_hour, max_hour, lead_times[i]))
    return times


"""def save_set(dataset, name: str):
    for i in range(len(dataset)):
        # Constants
        if i < 3:
            np.save(f"datasets/{name}/train_set_{i}.npy", dataset[i])
        # Singles
        if 2 < i < 7:
            for j in range(len(dataset[i])):
                np.save(f"datasets/{name}/train_set_{i}_{j}.npy", dataset[i][j].filled())
        if 6 < i < 13:
            for j in range(len(dataset[i])):
                np.save(f"datasets/{name}/train_set_{i}_{j}.npy", dataset[i][j])"""


def plot(name, resnet_rmse, resnet_acc, unet_rmse, unet_acc, vit_rmse, vit_acc):
    lead_time = [6, 24, 72, 120, 240]

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot RMSE on the first subplot
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
    for i in range(len(lists)):                          # Variable
        for j in range(5):                               # Net
            lists[i].append(stats[j][i])
    return lists


class PeriodicPadding2D(nn.Module):
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, inputs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = torch.cat((inputs[:, :, :, -self.pad_width:],
                                   inputs,
                                   inputs[:, :, :, :self.pad_width],), dim=-1,).to(device)
        # Zero padding in the lat direction
        inputs_padded = nn.functional.pad(inputs_padded, (0, 0, self.pad_width, self.pad_width)).to(device)
        return torch.permute(inputs_padded, (1, 0, 2, 3)).to(device)


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
        if pad_x.shape[1] != self.in_channels:
            pad_x = torch.permute(pad_x, (1, 0, 2, 3)).to(device)

        x = self.conv1(pad_x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x += residual

        return x


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
