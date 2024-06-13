from utils import *


class Baseline(nn.Module):
    def __init__(self, in_channels, out_channels, processor):
        super().__init__()
        self.in_channels = in_channels
        self.processor = processor
        self.periodic_zeros_padding = PeriodicPadding2D(3)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=0).to(device)
        self.set_climatology = []

    def __call__(self, data, six_hours_ago, twelve_hours_ago, target, constants, for_test=0):
        current_data = np.concatenate((constants, data), axis=1)
        pred = self.forward(np.concatenate((current_data, six_hours_ago, twelve_hours_ago), axis=1))
        if for_test:
            self.set_climatology.append(target)
        return pred

    def forward(self, x):
        x = self.processor.process(x)
        x = self.conv(self.periodic_zeros_padding(x))
        return x
