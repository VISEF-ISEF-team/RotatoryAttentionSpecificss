import torch.nn as nn
import torch
from .DynamicRotatoryAttention import DynamicRotatoryAttentionModule
from torchinfo import summary


class ConvolutionBatchNormblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        adaptive convolution + batch norm + activation block
        """

        super().__init__()

        # convolutional layer

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, padding=padding, kernel_size=kernel_size, stride=stride)

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, first=False):

        super().__init__()

        # conv block

        self.first = first

        # identity mapping
        self.identity = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0)

        if first == True:
            # initialize first time convolution

            self.single_first_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=stride)

            self.first_conv_block = ConvolutionBatchNormblock(
                in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same")

        else:
            self.conv_block_1 = ConvolutionBatchNormblock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)

            self.conv_block_2 = ConvolutionBatchNormblock(
                in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same")

        # activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # pass through convolutional block

        if self.first:
            x_ = self.single_first_conv(x)
            x_ = self.first_conv_block(x_)

            x = self.identity(x) + x_

            x = self.relu(x)

            return x

        else:
            x_ = self.conv_block_1(x)
            x_ = self.conv_block_2(x_)

            x = self.identity(x) + x_

            x = self.relu(x)

            return x


class RotatoryUpSampleBlock(nn.Module):
    def __init__(self, flattened_dim, window_size, in_channels, out_channels: list, kernel_size):

        super().__init__()

        self.flattened_dim = flattened_dim
        self.rot_inc = in_channels
        self.window_size = window_size

        # rotatory attention
        self.rotatory_layer_norm = nn.LayerNorm(
            normalized_shape=in_channels)

        self.rag = DynamicRotatoryAttentionModule(
            seq_length=flattened_dim, embed_dim=in_channels, window_size=window_size)

        # upsampling feature
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=2, padding=0)

        # convolutional blocks
        self.conv_block_1 = ConvolutionBatchNormblock(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same")

        self.conv_block_2 = ConvolutionBatchNormblock(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same")

        # identity mapping
        self.identity = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        # activation
        self.relu = nn.ReLU(inplace=True)

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

        # flattened_dim, inc

        # 8, flattned_dim, inc -> flattned_dim, inc -> normalise -> permute -> reshape

        context_list = torch.stack(context_list)
        context_list = self.rotatory_layer_norm(context_list)
        context_mean = torch.mean(context_list, dim=0)

        context_mean = context_mean.permute(1, 0)

        context_mean = context_mean.view(
            self.rot_inc, int(self.flattened_dim ** 0.5), int(self.flattened_dim ** 0.5))

        # add with the attention score turns this into an additional attention gate
        x = x + context_mean
        return x

    def forward(self, x, s):
        # upsample original path
        x = self.up(x)

        x = torch.concat((x, s), dim=1)

        x = self.apply_rotatory_attention(x)

        x_ = self.conv_block_1(x)
        x_ = self.conv_block_2(x_)

        x = self.identity(x) + x_

        x = self.relu(x)

        return x


class Rotatory_ResUNet_Attention(nn.Module):
    def __init__(self, inc, outc, image_size, window_size):

        super().__init__()

        # downward path
        model_kernel_size = 3

        self.e1 = DownSampleBlock(
            in_channels=inc, out_channels=64, stride=1, kernel_size=model_kernel_size, first=True)

        self.e2 = DownSampleBlock(
            in_channels=64, out_channels=128, stride=2, kernel_size=model_kernel_size)

        self.e3 = DownSampleBlock(
            in_channels=128, out_channels=256, stride=2, kernel_size=model_kernel_size)

        # bridge
        self.b1 = DownSampleBlock(
            in_channels=256, out_channels=512, stride=2, kernel_size=model_kernel_size)

        # upward path
        self.u1 = RotatoryUpSampleBlock(flattened_dim=int((image_size // (2 ** 2)) ** 2),
                                        window_size=window_size,
                                        in_channels=512, out_channels=256, kernel_size=model_kernel_size)

        self.u2 = RotatoryUpSampleBlock(flattened_dim=int((image_size // (2 ** 1)) ** 2),
                                        window_size=window_size,
                                        in_channels=256, out_channels=128, kernel_size=model_kernel_size)

        self.u3 = RotatoryUpSampleBlock(flattened_dim=int((image_size // (2 ** 0)) ** 2),
                                        window_size=window_size,
                                        in_channels=128, out_channels=64, kernel_size=model_kernel_size)

        # output
        self.out = nn.Conv2d(
            in_channels=64, out_channels=outc, kernel_size=1, stride=1)

    def forward(self, x):
        # down path
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)

        # bridge
        b1 = self.b1(x3)

        # up path
        u1 = self.u1(b1, x3)
        u2 = self.u2(u1, x2)
        u3 = self.u3(u2, x1)

        out = self.out(u3)

        return out


if __name__ == "__main__":

    device = torch.device("cuda")

    model = Rotatory_ResUNet_Attention(
        inc=3, outc=8, image_size=256, window_size=7).to(device=device)

    input = torch.rand(8, 3, 256, 256).to(device=device)

    output = model(input)

    print(f"Output: {output.shape}")
