import torch

from torch import nn, Tensor

SHAPE = (256, 256)


class baselineCAE(nn.Module):

    def __init__(self, color_mode):
        super().__init__()
        if color_mode == 'gray_scale':
            channels = 1
        elif color_mode == 'rgb':
            channels = 3

        img_dim = (*SHAPE, channels)

        encoding_dim = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3),
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ReflectionPad2d(padding=(1, 0, 1, 0)),
            nn.MaxPool2d((2, 2)),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ReflectionPad2d(padding=(1, 0, 1, 0)),
            nn.MaxPool2d((2, 2)),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ReflectionPad2d(padding=(1, 0, 1, 0)),
            nn.MaxPool2d((2, 2)),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ReflectionPad2d(padding=(1, 0, 1, 0)),
            nn.MaxPool2d((2, 2)),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ReflectionPad2d(padding=(1, 0, 1, 0)),
            nn.MaxPool2d((2, 2)),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ReflectionPad2d(padding=(1, 0, 1, 0)),
            nn.MaxPool2d((2, 2)),
        )
        hid_dim = 2048  # know later
        self.flatten = nn.Sequential(
            nn.Linear(hid_dim, encoding_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )

        hid_channel = 4  # know later: encoding_dim = 4 * 4 * hid_channel
        # decoder
        self.d_layer1 = nn.Sequential(
            nn.Conv2d(hid_channel, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.d_layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.d_layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.d_layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.d_layer5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.d_layer6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.d_layer7 = nn.Sequential(
            nn.Conv2d(32, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        n = x.size(0)
        print(x.size())
        x = self.layer1(x)
        print('layer 1', x.size())
        x1 = x.clone()
        x = self.layer2(x)
        print('layer 2', x.size())
        x = self.layer3(x)

        print('layer 3', x.size())
        x = self.layer4(x)
        print('layer 4', x.size())
        x = self.layer5(x)
        print('layer 5', x.size())
        x = self.layer6(x)

        print('hid size', x.size())
        x = x.view(n, -1)

        x = self.flatten(x)

        x = x.view(n, -1, 4, 4)
        x = self.d_layer1(x)
        print('layer 1', x.size())
        x = self.d_layer2(x)
        print('layer 2', x.size())
        x = self.d_layer3(x)
        print('layer 3', x.size())
        x = self.d_layer4(x)
        print('layer 4', x.size())
        x = self.d_layer5(x)
        print('layer 5', x.size())
        x2 = x.clone()
        x = self.d_layer6(x)
        print('layer 6', x.size())
        x = self.d_layer7(x)
        print('layer 7', x.size())

        outputs = {'e': x1,
                   'd': x2,
                   'o': x}
        return outputs


if __name__ == '__main__':
    m = baselineCAE(color_mode='grayscale')
    x = torch.randn(20, 1, 256, 256)
    y = m(x)
