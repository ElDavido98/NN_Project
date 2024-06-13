from ..Utils.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, processor, hidden_channels=64, channel_multiplications=(1, 2, 2),
                 blocks=2, use_attention=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.processor = processor
        self.set_climatology = []
        self.periodic_zeros_padding = PeriodicPadding2D(3)
        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=7, stride=1, padding=0).to(device)
        out_channels = in_channels = self.hidden_channels
        self.n_resolutions = len(channel_multiplications)
        self.blocks = blocks
        # Downward path
        self.down_blocks = []
        for i in range(self.n_resolutions):     # 3
            out_channels = in_channels * channel_multiplications[i]
            for _ in range(self.blocks):     # 2
                self.down_blocks.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < self.n_resolutions - 1:
                self.down_blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1).to(device))
        self.down_blocks = nn.ModuleList(self.down_blocks).to(device)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        ).to(device)
        # Upward path
        self.up_blocks = []
        for i in reversed(range(self.n_resolutions)):       # 3
            out_channels = in_channels
            for _ in range(self.blocks):     # 2
                self.up_blocks.append(ResidualBlock(in_channels+out_channels, out_channels))
            out_channels = in_channels // channel_multiplications[i]
            self.up_blocks.append(ResidualBlock(in_channels + out_channels, out_channels))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                self.up_blocks.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1).to(device))
        self.up_blocks = nn.ModuleList(self.up_blocks).to(device)
        self.norm = nn.BatchNorm2d(self.hidden_channels).to(device)
        self.leaky_relu = nn.LeakyReLU(0.3).to(device)
        self.out = nn.Conv2d(in_channels, self.out_channels, kernel_size=7, padding=0).to(device)

    def __call__(self, data, six_hours_ago, twelve_hours_ago, target, constants, for_test=0):
        current_data = np.concatenate((constants, data), axis=1)
        pred = self.forward(np.concatenate((current_data, six_hours_ago, twelve_hours_ago), axis=1))
        if for_test:
            self.set_climatology.append(target)
        return pred

    def forward(self, x):
        x = self.processor.process(x)
        x = self.conv1(self.periodic_zeros_padding(x))
        skips = [x]
        # Downward path
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
        # Bottleneck
        x = self.bottleneck(x)
        # Upward path
        for up_block in self.up_blocks:
            if isinstance(up_block, nn.ConvTranspose2d):
                x = up_block(x)
            else:
                skip_connection = skips.pop()
                x = torch.cat((x, skip_connection), dim=1).to(device)
                x = up_block(x)
        x = self.periodic_zeros_padding(x)
        x = self.out(self.leaky_relu(self.norm(x)))
        return x
