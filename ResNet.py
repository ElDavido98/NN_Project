import torch.nn as nn
import torch
from utils import PeriodicPadding2D, ResidualBlock, extractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, processor, hidden_channels=128, num_blocks=28):
        super(ResNet, self).__init__()
        self.periodic_zeros_padding = PeriodicPadding2D(3)
        self.image_projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=0).to(device)
        self.res_net_blocks = self._make_layer(ResidualBlock, hidden_channels, hidden_channels, num_blocks=num_blocks)
        self.norm = nn.BatchNorm2d(hidden_channels).to(device)
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=7, stride=1, padding=3).to(device)
        self.leaky_relu = nn.LeakyReLU(0.3).to(device)

        self.processor = processor
        self.set_climatology = []

    def __call__(self, data, low_year, max_year, lead_time, time):
        year, hour, year_6, hour_6, year_12, hour_12, year_targ, hour_targ = time

        dataset = extractor(data, year, hour, year_6, hour_6, year_12, hour_12, for_pred=True)

        pred = self.forward(dataset)
        target = self.processor.process(extractor(data, year_targ, hour_targ, for_pred=False))

        target = target.cpu()

        self.set_climatology.append(target.detach().numpy())

        return pred, target.detach().numpy()

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers).to(device)

    def forward(self, x):
        x = self.processor.process(x)
        x = self.image_projection(self.periodic_zeros_padding(x))
        x = self.res_net_blocks(x)
        x = self.out(self.leaky_relu(self.norm(x)))
        x = x.reshape([3, 1, 32, 64])
        x = x.cpu()
        return x.detach().numpy()
