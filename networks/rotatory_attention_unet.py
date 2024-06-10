import torch
import torch.nn as nn
from DynamicRotatoryAttention import DynamicRotatoryAttentionModule


class conv_block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = conv_block(inc, outc)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)

        return s, p


class decoder_block(nn.Module):
    def __init__(self, inc, outc, flattened_dim, window_size):
        super().__init__()

        self.inc = inc
        self.outc = outc
        self.flattened_dim = flattened_dim
        self.window_size = window_size

        self.up = nn.ConvTranspose2d(
            in_channels=inc, out_channels=outc, kernel_size=2, stride=2)

        self.rag = DynamicRotatoryAttentionModule(
            seq_length=flattened_dim, embed_dim=inc, window_size=window_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.c1 = conv_block(outc + outc, outc)

    def forward(self, x, s):
        # x and s must have same height and width
        """get left right vector from output"""
        n_sample = x.shape[0]
        reach = (self.window_size - 1) // 2

        new_output = torch.empty(size=x.shape).to(x.device)

        # add leftover inputs from left side
        for i in range(0, 0 + reach + 1, 1):
            new_output[i] = x[i]

        # add leftover inputs from right side
        for i in range(n_sample - reach, n_sample, 1):
            new_output[i] = x[i]

        for i in range(0 + reach, n_sample - reach, 1):
            target = x[i].view(self.inc, -1).permute(1, 0)

            left_list = [x[i].view(self.inc, -1).permute(1, 0)
                         for i in range(i-reach, i)]
            right_list = [x[i].view(self.inc, -1).permute(1, 0)
                          for i in range(i + 1, i + reach + 1)]

            f_list = left_list + right_list

            output = self.rag(target, f_list)

            output = torch.unsqueeze(output.view(
                self.inc, int(self.flattened_dim ** 0.5), int(self.flattened_dim ** 0.5)), dim=0)

            new_output[i] = output

        x = self.up(new_output)

        ##########

        """Improve upon unet attention"""
        x = self.relu(x + s)
        x = self.sigmoid(x)

        ##########

        x = torch.concat([x, s], dim=1)
        x = self.c1(x)
        return x


class Rotatory_Attention_Unet(nn.Module):
    def __init__(self, inc, outc, image_size, window_size):
        super().__init__()

        self.e1 = encoder_block(inc, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        self.b1 = conv_block(256, 512)

        self.d1 = decoder_block(512, 256, int(
            image_size // (2 ** 3)) ** 2, window_size=window_size)
        self.d2 = decoder_block(256, 128, int(
            image_size // (2 ** 2)) ** 2, window_size=window_size)
        self.d3 = decoder_block(128, 64, int(
            image_size // (2 ** 1)) ** 2, window_size=window_size)

        self.output = nn.Conv2d(64, outc, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)

        return output


if __name__ == "__main__":
    device = torch.device("cuda")

    model = Rotatory_Attention_Unet(
        inc=3, outc=8, image_size=256, window_size=5).to(device)

    x = torch.rand(8, 3, 256, 256).to(device)

    output = model(x)

    print(f"Output: {output.shape}")
