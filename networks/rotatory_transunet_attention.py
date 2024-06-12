import torch
import torch.nn as nn
from .vit import PatchEmbedding
from .DynamicRotatoryAttention import DynamicRotatoryAttentionModule


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
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        kernel_size = kernel_size
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


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, kernel_size, features: list):
        super().__init__()

        # first convolution
        self.d1 = DownSampleBlock(
            in_channels=in_channels, out_channels=features[0], kernel_size=kernel_size)

        self.d2 = DownSampleBlock(
            in_channels=64, out_channels=features[1], kernel_size=kernel_size)

        self.d3 = DownSampleBlock(
            in_channels=128, out_channels=features[2], kernel_size=kernel_size)

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


class CUPRotatoryUpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, flattened_dim, window_size):
        super().__init__()

        self.window_size = window_size
        self.flattened_dim = flattened_dim
        self.rot_inc = in_channels
        kernel_size = kernel_size
        padding = "same"
        stride = 1

        self.up = nn.Upsample(scale_factor=2)

        self.single_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding="same")

        self.conv_block = ConvolutionBatchNormblock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        # rotatory attention
        self.rotatory_layer_norm = nn.LayerNorm(
            normalized_shape=in_channels)

        self.rag = DynamicRotatoryAttentionModule(
            seq_length=flattened_dim, embed_dim=in_channels, window_size=window_size)

    def apply_rotatory_attention(self, x):
        n_sample = x.shape[0]

        reach = (self.window_size - 1) // 2

        assert self.window_size < n_sample, print(
            f"Batch does not contain enough image for rotatory attention.")

        context_list = []

        for i in range(0 + reach, n_sample - reach, 1):
            # (num_features, hxw) -> (hxw, features)
            target = x[i].view(self.rot_inc, -1).permute(1, 0)

            left_list = [x[i].view(self.rot_inc, -1).permute(1, 0)
                         for i in range(i-reach, i)]
            right_list = [x[i].view(self.rot_inc, -1).permute(1, 0)
                          for i in range(i + 1, i + reach + 1)]

            f_list = left_list + right_list

            output = self.rag(target, f_list)

            context_list.append(output)

        context_list = torch.stack(context_list)
        context_list = self.rotatory_layer_norm(context_list)
        context_mean = torch.mean(context_list, dim=0)

        context_mean = context_mean.permute(1, 0)

        context_mean = context_mean.view(
            self.rot_inc, int(self.flattened_dim ** 0.5), int(self.flattened_dim ** 0.5))

        # add with the attention score turns this into an additional attention gate
        x = x + context_mean
        return x

    def forward(self, x, skip):
        x = self.up(x)
        x = self.single_conv(x)

        x = torch.cat((x, skip), dim=1)

        x = self.apply_rotatory_attention(x)

        x = self.conv_block(x)

        return x


class CNNDecoder(nn.Module):
    def __init__(self, patch_size, kernel_size, image_size, embedding_dim, window_size, features, cup=True):
        super().__init__()

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.window_size = window_size
        self.features = features
        self.cup = cup

        # single convolution for converting channel
        self.single_conv = nn.Conv2d(
            in_channels=embedding_dim, out_channels=512, kernel_size=3, stride=1, padding="same")

        # create empty decoder blocks
        self.u1 = CUPRotatoryUpSampleBlock(
            in_channels=self.features[-1], out_channels=self.features[-2], kernel_size=self.kernel_size, flattened_dim=int((image_size * 2 * 1) ** 2), window_size=window_size)

        self.u2 = CUPRotatoryUpSampleBlock(
            in_channels=self.features[-2], out_channels=self.features[-3], kernel_size=self.kernel_size, flattened_dim=int((image_size * 2 * 2) ** 2), window_size=window_size)

        self.u3 = CUPRotatoryUpSampleBlock(
            in_channels=self.features[-3], out_channels=self.features[-4], kernel_size=self.kernel_size, flattened_dim=int((image_size * 2 * 4) ** 2), window_size=window_size)

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
        size = self.image_size * 2

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


class Rotatory_TransUNet_Attention(nn.Module):
    def __init__(self, inc, outc, image_size, window_size):
        super().__init__()

        kernel_size = 3
        features = [64, 128, 256, 512]
        recalculated_patch_size = 2

        # encoder
        self.encoder = CNNEncoder(
            in_channels=inc, kernel_size=kernel_size, features=features)

        # transformer

        # patch size = 2 is fixed if you want the output of the ViT component to be (H / 16, W / 16) after 3 rounds of downsampling
        self.vit = ViTComponent(img_size=image_size // (2 ** 3), inc=256,
                                patch_size=recalculated_patch_size, embedding_dim=768)

        # calculate new image size after ViT component
        new_image_size = int(image_size // (2 ** 3) //
                             recalculated_patch_size)

        # decoder
        self.decoder = CNNDecoder(patch_size=self.vit.patch_size,
                                  kernel_size=kernel_size,
                                  image_size=new_image_size,
                                  embedding_dim=self.vit.embedding_dim, window_size=window_size,
                                  features=features)

        # classifer
        self.output = nn.Conv2d(
            in_channels=features[0], out_channels=outc, kernel_size=1, stride=1, padding="same")

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
    a = torch.rand(8, 1, 224, 224)

    model = Rotatory_TransUNet_Attention(
        inc=1, outc=12, image_size=224, window_size=5)

    output = model(a)

    print(f"Output: {output.shape}")
