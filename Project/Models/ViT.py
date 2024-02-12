import torch
import torch.nn as nn
from timm.layers import PatchEmbed, DropPath
from timm.models.vision_transformer import Block
from ..Utils.utils import extractor, return_to_image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViT(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, processor, patch_size=2, embedding_dim=128, depth=8,
                 num_heads=4, mlp_ratio=4, prediction_depth=2, drop_path_rate=0.1, dropout_rate=0.1,
                 learn_positional_embedding=False):
        super(ViT, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.processor = processor
        self.set_climatology = []
        # Patch Embedding
        self.patch_embedding = PatchEmbed(self.img_size, patch_size, self.in_channels, embedding_dim).to(device)
        self.num_patches = self.patch_embedding.num_patches
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim).to(device),
                                      requires_grad=learn_positional_embedding).to(device)

        self.pos_drop = nn.Dropout(dropout_rate).to(device)
        dpr = [prob.item() for prob in torch.linspace(0, drop_path_rate, depth).to(device)]

        # Encoder (Transformer encoder)
        self.encoder = nn.ModuleList(
            [Block(
                embedding_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                drop_path=dpr[i],
                proj_drop=dropout_rate,
                attn_drop=dropout_rate)
                for i in range(depth)
            ]
        ).to(device)        # norm_layer=nn.LayerNorm,

        self.norm = nn.LayerNorm(embedding_dim).to(device)

        # Prediction head (MLP Head)
        prediction_layers = []
        for _ in range(prediction_depth):
            prediction_layers.append(nn.Linear(embedding_dim, embedding_dim).to(device))
            prediction_layers.append(nn.LeakyReLU().to(device))
        prediction_layers.append(nn.Linear(embedding_dim, out_channels * patch_size ** 2).to(device))
        self.prediction_head = nn.Sequential(*prediction_layers).to(device)

    def __call__(self, data, low_year, max_year, lead_time, time):
        year, hour, year_6, hour_6, year_12, hour_12, year_targ, hour_targ = time

        dataset = extractor(data, year, hour, year_6, hour_6, year_12, hour_12, for_pred=True)

        pred = self.forward(dataset)
        target = self.processor.process(extractor(data, year_targ, hour_targ, for_pred=False))

        target = target.cpu()

        self.set_climatology.append(target.detach().numpy())

        return pred, target.detach().numpy()

    def forward(self, x):
        x = self.processor.process(x)
        # Patch embedding
        x = self.patch_embedding(torch.permute(x, (1, 0, 2, 3)).to(device))
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Encoder
        for block in self.encoder:
            x = block.forward(x)
        # x = self.encoder.forward(x)
        x = self.norm(x)
        # Prediction head
        x = self.prediction_head(x)
        x = return_to_image(x, self.patch_size, self.out_channels, self.img_size)
        x = x.cpu()
        return x.detach().numpy()

    def _init_net(self, embedding_dim, n=10000):
        pos_embed = torch.zeros(1, self.num_patches, embedding_dim).to(device)
        for i in range(1):
            for j in range(self.num_patches):
                for k in range(int(embedding_dim / 2)):
                    denominator = np.power(n, 2 * i / embedding_dim)
                    pos_embed[i, k, 2 * i] = np.sin(k / denominator)
                    pos_embed[i, k, 2 * i + 1] = np.cos(k / denominator)

        return pos_embed
