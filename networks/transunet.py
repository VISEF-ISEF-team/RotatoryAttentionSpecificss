import torch
import torch.nn as nn
from vit import PatchEmbedding


class ConvolutionBatchNormblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        adaptive convolution + batch norm + activation block
        """
        super().__init__()

        # convolutional layer

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, padding=padding, kernel_size=kernel_size, stride=stride, bias=False)

        # batch norm
        self.bn = nn.BatchNorm2d(num_features=in_channels)

        # activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        kernel_size = 3
        padding = "same"
        stride = 1

        self.conv_block_1 = ConvolutionBatchNormblock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        self.conv_block_2 = ConvolutionBatchNormblock(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.pool(x)

        return x


class CUPUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        kernel_size = 3
        padding = "same"
        stride = 1

        self.up = nn.Upsample(scale_factor=2)

        self.single_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

        self.conv_block = ConvolutionBatchNormblock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.single_conv(x)

        x = torch.cat((x, skip), dim=1)

        x = self.conv_block(x)

        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # first convolution
        self.d1 = DownSampleBlock(in_channels=in_channels, out_channels=64)
        self.d2 = DownSampleBlock(in_channels=64, out_channels=128)
        self.d3 = DownSampleBlock(in_channels=128, out_channels=256)

    def forward(self, x):
        x1 = self.d1(x)

        x2 = self.d2(x1)

        x3 = self.d3(x2)

        x4 = x3.clone()

        return x4, (x3, x2, x1)


class ViTComponent(nn.Module):
    def __init__(self, img_size: int = 256,
                 inc: int = 1,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 256,
                 mlp_size: int = 3072,
                 num_heads: int = 16,
                 mlp_dropout: int = 0.1,
                 embedding_dropout: int = 0.1):
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisble by patch size."

        # store public variables for access from other classes
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # calculate number of patches
        self.num_patches = (img_size * img_size) // (patch_size ** 2)

        # create learnable position embeddings
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches, embedding_dim))

        # create embeddings dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # patch embedding layer
        self.patch_embedding = PatchEmbedding(
            inc=inc, patch_size=patch_size, embedding_dim=embedding_dim)

        # create transformer encoder block
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim, dim_feedforward=mlp_size, nhead=num_heads, dropout=mlp_dropout, norm_first=True), num_layers=num_transformer_layers, enable_nested_tensor=False)

    def forward(self, x):
        # create the patch embedding
        x = self.patch_embedding(x)  # -> (batch_size, num_patches, embed_dim)

        # add positional embedding
        x = self.position_embedding + x

        # apply embedding dropout
        x = self.embedding_dropout(x)

        # pass position and patch embedding to transformer encoder
        x = self.transformer_encoder(x)

        return x


class CNNDecoder(nn.Module):
    def __init__(self, patch_size, embedding_dim, cup=True):
        super().__init__()

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # single convolution for converting channel
        self.single_conv = nn.Conv2d(
            in_channels=embedding_dim, out_channels=512, kernel_size=3, stride=1, padding="same")

        # decoder blocks
        self.u1 = CUPUpsampleBlock(in_channels=512, out_channels=256)

        self.u2 = CUPUpsampleBlock(in_channels=256, out_channels=128)

        self.u3 = CUPUpsampleBlock(in_channels=128, out_channels=64)

        self.u4 = nn.Upsample(scale_factor=2)

    def transformer_reshape(self, x):
        # (batch, seq_length, dim) -> (batch, dim, h / p, w / p)

        batch_size = x.shape[0]
        dimension = x.shape[2]
        flattened = x.shape[1]

        assert dimension == self.embedding_dim, print(
            f"Mismatch in dimension from input and dimension received as arguments.")

        # reshape dimension
        x = x.permute(0, 2, 1).contiguous()

        # calculate height and width
        size = int((flattened * (self.patch_size ** 2)) ** 0.5)

        x = x.view(batch_size, dimension, size //
                   self.patch_size, size // self.patch_size)

        x = self.single_conv(x)

        return x

    def forward(self, x, skip_connections: list = None):
        x = self.transformer_reshape(x)

        u1 = self.u1(x, skip_connections[0])
        u2 = self.u2(u1, skip_connections[1])
        u3 = self.u3(u2, skip_connections[2])

        u4 = self.u4(u3)

        return u4


class TransUNet(nn.Module):
    def __init__(self, inc, outc, image_size):
        super().__init__()

        # encoder
        self.encoder = CNNEncoder(in_channels=inc)

        # transformer
        self.vit = ViTComponent(img_size=image_size // (2 ** 3), inc=256,
                                patch_size=2, embedding_dim=768)

        # decoder
        self.decoder = CNNDecoder(patch_size=self.vit.patch_size,
                                  embedding_dim=self.vit.embedding_dim)

        # classifer
        self.output = nn.Conv2d(
            in_channels=64, out_channels=outc, kernel_size=1, stride=1, padding="same")

    def forward(self, x):
        # pass through encoder and get skip connections
        x, skip_connections = self.encoder(x)

        # pass through vision transformer
        x = self.vit(x)

        # pass through decoder
        x = self.decoder(x, skip_connections)

        # output
        x = self.output(x)

        return x


if __name__ == "__main__":
    a = torch.rand(8, 1, 256, 256)

    model = TransUNet(inc=1, outc=8, image_size=256)

    output = model(a)

    print(f"Output: {output.shape}")
