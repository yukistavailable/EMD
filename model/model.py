import os
import torch
import torch.nn as nn
from torchinfo import summary
import torchvision.utils as vutils


class EMD(nn.Module):
    def __init__(
            self,
            content_input_nc=1,
            style_input_nc=1,
            ngf=64,
            gpu_id='cuda',
    ):
        super(EMD, self).__init__()
        self.content_input_nc = content_input_nc
        self.style_input_nc = style_input_nc
        self.ngf = ngf
        self.gpu_id = gpu_id
        self.content_encoder = Encoder(
            self.content_input_nc, self.ngf, is_content=True)
        self.style_encoder = Encoder(
            self.style_input_nc, self.ngf, is_content=False)
        self.decoder = Decoder()

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['style_encoder', 'decoder']:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(
                    '[Network %s] Total number of parameters : %.3f M' %
                    (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, save_dir, epoch):
        save_file_name = f'{epoch}.pth'
        save_path = os.path.join(save_dir, save_file_name)
        torch.save(self.state_dict(), save_path)

    def sample(self, content, style, basename):
        count = 0
        with torch.no_grad():
            output = self.forward(content, style)
            if output is None:
                return None
            for _output in output:
                vutils.save_image(
                    _output,
                    os.path.join(basename, f'{count}.png')
                )
            count += 1
        return output

    def forward(self, content, style):
        c1, c2, c3, c4, c5, c6, c7, _content = self.content_encoder(content)
        _style = self.style_encoder(style)
        x = self.decoder(_content, _style, c1, c2, c3, c4, c5, c6, c7)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            input_nc=1,
            ngf=64,
            is_content=True,
    ):
        """
        Args:
            input_nc (int): the number of channels in input images
            output_nc (int): the number of channels in output images
            num_downs (int): the number of downsamplings
            ngf (int): the number of filters in the last conv layer
        """
        super(Encoder, self).__init__()

        self.is_content = is_content
        self.down_relu = nn.LeakyReLU(0.2)
        self.norm2 = nn.BatchNorm2d(ngf)
        self.norm3 = nn.BatchNorm2d(ngf * 2)
        self.norm4 = nn.BatchNorm2d(ngf * 4)
        self.norm5 = nn.BatchNorm2d(ngf * 8)

        self.outer_down_block = nn.Conv2d(
            in_channels=input_nc,
            out_channels=ngf,
            kernel_size=5,
            stride=1)
        self.down_block1 = nn.Conv2d(
            in_channels=ngf,
            out_channels=ngf * 2,
            kernel_size=3,
            stride=2)
        self.down_block2 = nn.Conv2d(
            in_channels=ngf * 2,
            out_channels=ngf * 4,
            kernel_size=3,
            stride=2)
        self.down_block3 = nn.Conv2d(
            in_channels=ngf * 4,
            out_channels=ngf * 8,
            kernel_size=3,
            stride=2)
        self.down_block4 = nn.Conv2d(
            in_channels=ngf * 8,
            out_channels=ngf * 8,
            kernel_size=3,
            stride=2)
        self.down_block5 = nn.Conv2d(
            in_channels=ngf * 8,
            out_channels=ngf * 8,
            kernel_size=3,
            stride=2)
        self.down_block6 = nn.Conv2d(
            in_channels=ngf * 8,
            out_channels=ngf * 8,
            kernel_size=3,
            stride=2)
        self.down_block7 = nn.Conv2d(
            in_channels=ngf * 8,
            out_channels=ngf * 8,
            padding=1,
            kernel_size=3,
            stride=2)

    def forward(self, x):
        """
        Args:
            x
        """
        c1 = self.down_relu(self.norm2(self.outer_down_block(x)))
        c2 = self.down_relu(self.norm3(self.down_block1(c1)))
        c3 = self.down_relu(self.norm4(self.down_block2(c2)))
        c4 = self.down_relu(self.norm5(self.down_block3(c3)))
        c5 = self.down_relu(self.norm5(self.down_block4(c4)))
        c6 = self.down_relu(self.norm5(self.down_block5(c5)))
        c7 = self.down_relu(self.norm5(self.down_block6(c6)))
        c8 = self.down_relu(self.norm5(self.down_block7(c7)))
        if self.is_content:
            return c1, c2, c3, c4, c5, c6, c7, c8
        return c8


class Decoder(nn.Module):
    def __init__(
            self,
            output_nc=1,
            ngf=64,):
        super(Decoder, self).__init__()

        self.norm1 = nn.BatchNorm2d(output_nc)
        self.norm2 = nn.BatchNorm2d(ngf)
        self.norm3 = nn.BatchNorm2d(ngf * 2)
        self.norm4 = nn.BatchNorm2d(ngf * 4)
        self.norm5 = nn.BatchNorm2d(ngf * 8)

        self.up_relu = nn.ReLU()
        self.up_norm = nn.BatchNorm2d(output_nc)

        self.mixer = nn.Bilinear(
            in1_features=ngf * 8,
            in2_features=ngf * 8,
            out_features=ngf * 8)

        self.up_block1 = nn.ConvTranspose2d(
            in_channels=ngf * 8 * 2,
            out_channels=ngf * 8,
            stride=1,
            kernel_size=2,
        )
        self.up_block2 = nn.ConvTranspose2d(
            in_channels=ngf * 8 * 2,
            out_channels=ngf * 8,
            stride=2,
            kernel_size=3,
            output_padding=1,
        )
        self.up_block3 = nn.ConvTranspose2d(
            in_channels=ngf * 8 * 2,
            out_channels=ngf * 8,
            stride=2,
            kernel_size=3,
            output_padding=1,
        )
        self.up_block4 = nn.ConvTranspose2d(
            in_channels=ngf * 8 * 2,
            out_channels=ngf * 8,
            stride=2,
            kernel_size=3,
            output_padding=1,
        )
        self.up_block5 = nn.ConvTranspose2d(
            in_channels=ngf * 8 * 2,
            out_channels=ngf * 4,
            stride=2,
            kernel_size=3,
            output_padding=1,
        )
        self.up_block6 = nn.ConvTranspose2d(
            in_channels=ngf * 4 * 2,
            out_channels=ngf * 2,
            stride=2,
            kernel_size=3,
            output_padding=0,
        )
        self.up_block7 = nn.ConvTranspose2d(
            in_channels=ngf * 2 * 2,
            out_channels=ngf,
            stride=2,
            kernel_size=3,
            output_padding=1,
        )
        self.outer_up_block = nn.ConvTranspose2d(
            in_channels=ngf * 2,
            out_channels=output_nc,
            stride=1,
            kernel_size=5,
        )

    def forward(self, content, style, c1, c2, c3, c4, c5, c6, c7):
        """
        Args:
            content
            style
        """
        batch_size, _, _, _ = content.shape
        batch_size_, _, _, _ = style.shape
        # assert batch_size == batch_size_, f"batch size of content and style must be equal. {batch_size} != {batch_size_}"
        if batch_size != batch_size_:
            return None

        _style = torch.reshape(style, (batch_size, 512))
        _content = torch.reshape(content, (batch_size, 512))
        x = self.mixer(_style, _content)
        x = torch.reshape(x, (batch_size, 512, 1, 1))
        x = self.up_relu(self.norm5(self.up_block1(
            torch.cat([x, content], dim=1))))
        x = self.up_relu(self.norm5(self.up_block2(
            torch.cat((x, c7), dim=1))))
        x = self.up_relu(self.norm5(self.up_block3(
            torch.cat([x, c6], dim=1))))
        x = self.up_relu(self.norm5(self.up_block4(
            torch.cat([x, c5], dim=1))))
        x = self.up_relu(self.norm4(self.up_block5(
            torch.cat([x, c4], dim=1))))
        x = self.up_relu(self.norm3(self.up_block6(
            torch.cat([x, c3], dim=1))))
        x = self.up_relu(self.norm2(self.up_block7(
            torch.cat([x, c2], dim=1))))
        x = self.up_relu(self.norm1(self.outer_up_block(
            torch.cat([x, c1], dim=1))))
        return x
