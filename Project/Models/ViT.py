from timm.layers import PatchEmbed, DropPath
from ..Utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViT(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, processor, patch_size=2, embedding_dim=128, depth=8,
                 num_heads=4, hidden_dimension=128, mlp_ratio=4, prediction_depth=2, drop_path_rate=0.1,
                 dropout_rate=0.1, learn_positional_embedding=True):
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
        self.pos_embed = nn.Parameter(self.trigonometric_pos(embedding_dim),
                                      requires_grad=learn_positional_embedding).to(device)
        self.pos_drop = nn.Dropout(dropout_rate).to(device)
        self.vit_block = []
        for _ in range(depth):
            self.vit_block.append(ViTBlock(embedding_dim, num_heads, mlp_ratio, dropout_rate, drop_path_rate))
        self.vit_block = nn.ModuleList(self.vit_block).to(device)
        # Prediction head (MLP Head)
        prediction_layers = []
        for _ in range(prediction_depth):
            prediction_layers.append(nn.Linear(hidden_dimension, hidden_dimension).to(device))
            prediction_layers.append(nn.LeakyReLU().to(device))
        prediction_layers.append(nn.Linear(hidden_dimension, out_channels * patch_size ** 2).to(device))
        self.prediction_head = nn.Sequential(*prediction_layers).to(device)

    def __call__(self, data, six_hours_ago, twelve_hours_ago, target, constants, for_test=0):
        current_data = np.concatenate((constants, data), axis=1)
        pred = self.forward(np.concatenate((current_data, six_hours_ago, twelve_hours_ago), axis=1))
        if for_test:
            self.set_climatology.append(target)
        return pred

    def forward(self, x):
        x = self.processor.process(x)
        # Patch embedding
        x = self.patch_embedding(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Encoder
        for vit_block in self.vit_block:
            x = vit_block.forward(x)
        # Prediction head
        x = self.prediction_head(x)
        x = return_to_image(x, self.patch_size, self.out_channels, self.img_size)
        return x

    def trigonometric_pos(self, embedding_dim, n=10000):
        pos_embed = torch.zeros(1, self.num_patches, embedding_dim).to(device)
        for i in range(1):
            for j in range(self.num_patches):
                for k in range(int(embedding_dim / 2)):
                    denominator = np.power(n, 2 * i / embedding_dim)
                    pos_embed[i, k, 2 * i] = np.sin(k / denominator)
                    pos_embed[i, k, 2 * i + 1] = np.cos(k / denominator)
        return pos_embed


class ViTBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio, dropout_rate, drop_path_rate):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim).to(device)

        self.query = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.key = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.value = nn.Linear(embedding_dim, embedding_dim).to(device)

        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate).to(device)
        if drop_path_rate > 0:
            self.drop_path1 = DropPath(drop_path_rate)
        else:
            self.drop_path1 = nn.Identity().to(device)

        self.norm2 = nn.LayerNorm(embedding_dim).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim * mlp_ratio)).to(device),
            nn.LeakyReLU().to(device),
            nn.Dropout(dropout_rate).to(device),
            nn.Linear(int(embedding_dim * mlp_ratio), embedding_dim).to(device),
            nn.LeakyReLU().to(device),
            nn.Dropout(dropout_rate).to(device)
        ).to(device)
        if drop_path_rate > 0:
            self.drop_path2 = DropPath(drop_path_rate)
        else:
            self.drop_path2 = nn.Identity().to(device)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.norm1(x)
        y = self.mha(query=self.query(x), key=self.key(x), value=self.value(x))
        x = x + self.drop_path1(y[0])
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
