from ..Utils.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, processor, hidden_channels=128, num_blocks=28):
        super(ResNet, self).__init__()
        self.periodic_zeros_padding = PeriodicPadding2D(3)
        self.image_projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=0).to(device)
        self.res_net_blocks = make_layer(ResidualBlock, hidden_channels, hidden_channels, num_blocks=num_blocks, change=1)
        self.norm = nn.BatchNorm2d(hidden_channels).to(device)      
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=7, stride=1, padding=3).to(device)
        self.leaky_relu = nn.LeakyReLU(0.3).to(device)      
        self.processor = processor
        self.set_climatology = []

    def __call__(self, data, six_hours_ago, twelve_hours_ago, target, constants, for_test=0):
        current_data = np.concatenate((constants, data), axis=1)
        pred = self.forward(np.concatenate((current_data, six_hours_ago, twelve_hours_ago), axis=1))
        if for_test:
            self.set_climatology.append(target)
        return pred

    def forward(self, x):
        x = self.processor.process(x)
        x = self.image_projection(self.periodic_zeros_padding(x))
        x = self.res_net_blocks(x)
        x = self.out(self.leaky_relu(self.norm(x)))
        return x

