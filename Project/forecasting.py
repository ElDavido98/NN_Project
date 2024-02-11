import torch.nn as nn
import torch
from .Utils.data_processing import *
from .Utils.metrics import latitude_weighting_function
from .Utils.utils import latitude_coordinates, define_times, printProgressBar
from .Models.network import Network
from .Models.ResNet import ResNet
from .Models.UNet import UNet
from .Models.ViT import ViT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


low_year_train, max_year_train = 0, 6
low_hour_train, max_hour_train = 0, 8753
low_year_val, max_year_val = 0, 0
low_hour_val, max_hour_val = 0, 4379


class Forecasting(nn.Module):
    def __init__(self, train_data, validation_data):
        super(Forecasting, self).__init__()
        self.device = device
        self.num_channels = 141
        self.out_channels = 3
        self.img_size = (self.height, self.width) = (32, 64)
        self.processor = PreProcessing(train_data)
        self.latitude_coordinates = latitude_coordinates

        # Nets
        res = ResNet(self.num_channels, self.out_channels, self.processor)
        u = UNet(self.num_channels, self.out_channels, self.processor)
        vit = ViT(self.num_channels, self.out_channels, self.img_size, self.processor)

        lat_weights = latitude_weighting_function(self.latitude_coordinates)

        # ResNets
        self.ResNet_6 = Network(res, 6, train_data, validation_data, lat_weights, 'ResNet')
        self.ResNet_24 = Network(res, 24, train_data, validation_data, lat_weights, 'ResNet')
        self.ResNet_72 = Network(res, 72, train_data, validation_data, lat_weights, 'ResNet')
        self.ResNet_120 = Network(res, 120, train_data, validation_data, lat_weights, 'ResNet')
        self.ResNet_240 = Network(res, 240, train_data, validation_data, lat_weights, 'ResNet')

        # UNets
        self.UNet_6 = Network(u, 6, train_data, validation_data, lat_weights, 'UNet')
        self.UNet_24 = Network(u, 24, train_data, validation_data, lat_weights, 'UNet')
        self.UNet_72 = Network(u, 72, train_data, validation_data, lat_weights, 'UNet')
        self.UNet_120 = Network(u, 120, train_data, validation_data, lat_weights, 'UNet')
        self.UNet_240 = Network(u, 240, train_data, validation_data, lat_weights, 'UNet')

        # ViTs
        self.ViT_6 = Network(vit, 6, train_data, validation_data, lat_weights, 'ViT')
        self.ViT_24 = Network(vit, 24, train_data, validation_data, lat_weights, 'ViT')
        self.ViT_72 = Network(vit, 72, train_data, validation_data, lat_weights, 'ViT')
        self.ViT_120 = Network(vit, 120, train_data, validation_data, lat_weights, 'ViT')
        self.ViT_240 = Network(vit, 240, train_data, validation_data, lat_weights, 'ViT')

    def train_forecasters(self, epochs=50, batch_size=128):
        self.ResNet_6.reset(), self.ResNet_24.reset(), self.ResNet_72.reset(), self.ResNet_120.reset(), self.ResNet_240.reset()
        self.UNet_6.reset(), self.UNet_24.reset(), self.UNet_72.reset(), self.UNet_120.reset(), self.UNet_240.reset()
        self.ViT_6.reset(), self.ViT_24.reset(), self.ViT_72.reset(), self.ViT_120.reset(), self.ViT_240.reset()

        print("Start Training")
        for epoch in range(epochs):
            print("Epoch ", epoch)
            self.ResNet_6.pre_steps(), self.ResNet_24.pre_steps(), self.ResNet_72.pre_steps(), self.ResNet_120.pre_steps(), self.ResNet_240.pre_steps()
            self.UNet_6.pre_steps(), self.UNet_24.pre_steps(), self.UNet_72.pre_steps(), self.UNet_120.pre_steps(), self.UNet_240.pre_steps()
            self.ViT_6.pre_steps(), self.ViT_24.pre_steps(), self.ViT_72.pre_steps(), self.ViT_120.pre_steps(), self.ViT_240.pre_steps()
            for _ in range(batch_size):
                # time_x = [year, hour, year_6, hour_6, year_12, hour_12, year_targ, hour_targ]
                lead_times = [6, 24, 72, 120, 240]
                time_6, time_24, time_72, time_120, time_240 = define_times(low_year_train, max_year_train,
                                                                            low_hour_train, max_hour_train, lead_times)
                # ResNets
                self.ResNet_6.step(time_6), self.ResNet_24.step(time_24), self.ResNet_72.step(time_72), self.ResNet_120.step(time_120), self.ResNet_240.step(time_240)
                # UNets
                self.UNet_6.step(time_6), self.UNet_24.step(time_24), self.UNet_72.step(time_72), self.UNet_120.step(time_120), self.UNet_240.step(time_240)
                # ViTs
                self.ViT_6.step(time_6), self.ViT_24.step(time_24), self.ViT_72.step(time_72), self.ViT_120.step(time_120), self.ViT_240.step(time_240)

            lead_times = [6, 24, 72, 120, 240]
            time_6, time_24, time_72, time_120, time_240 = define_times(low_year_val, max_year_val, low_hour_val,
                                                                        max_hour_val, lead_times)
            # ResNets
            done_Res_6 = self.ResNet_6.post_steps(time_6, epoch)
            done_Res_24 = self.ResNet_24.post_steps(time_24, epoch)
            done_Res_72 = self.ResNet_72.post_steps(time_72, epoch)
            done_Res_120 = self.ResNet_120.post_steps(time_120, epoch)
            done_Res_240 = self.ResNet_240.post_steps(time_240, epoch)
            # UNets
            done_U_6 = self.UNet_6.post_steps(time_6, epoch)
            done_U_24 = self.UNet_24.post_steps(time_24, epoch)
            done_U_72 = self.UNet_72.post_steps(time_72, epoch)
            done_U_120 = self.UNet_120.post_steps(time_120, epoch)
            done_U_240 = self.UNet_240.post_steps(time_240, epoch)
            # ViTs
            done_ViT_6 = self.ViT_6.post_steps(time_6, epoch)
            done_ViT_24 = self.ViT_24.post_steps(time_24, epoch)
            done_ViT_72 = self.ViT_72.post_steps(time_72, epoch)
            done_ViT_120 = self.ViT_120.post_steps(time_120, epoch)
            done_ViT_240 = self.ViT_240.post_steps(time_240, epoch)

            if all([done_Res_6, done_Res_24, done_Res_72, done_Res_120, done_Res_240, done_U_6, done_U_24, done_U_72,
                    done_U_120, done_U_240, done_ViT_6, done_ViT_24, done_ViT_72, done_ViT_120, done_ViT_240]):
                print("Stopped prematurely due to EarlyStopping")
                break
        print("End Training")

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
