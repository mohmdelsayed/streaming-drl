import torch
import logging

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter
from torch.distributions import Normal
from rl_suite.mlp_policies import orthogonal_weight_init


def random_augment(images, rad_height, rad_width):
    """ RAD from Laskin et al.,

    Args:
        images:
        rad_height:
        rad_width:

    Returns:

    """
    n, c, h, w = images.shape
    _h = h - 2 * rad_height
    _w = w - 2 * rad_width
    w1 = torch.randint(0, rad_width + 1, (n,))
    h1 = torch.randint(0, rad_height + 1, (n,))
    cropped_images = torch.empty((n, c, _h, _w), device=images.device).float()
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped_images[i][:] = image[:, h11:h11 + _h, w11:w11 + _w]
    return cropped_images

def conv_out_size(input_size, kernel_size, stride, padding=0):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.contiguous().view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class SSEncoderModel(nn.Module):
    """Convolutional encoder of pixels observations. Uses Spatial Softmax"""

    def __init__(self, image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax=False):
        super().__init__()
        self.spatial_softmax = spatial_softmax
        assert image_shape[-1] != 0, "This is an image encoder! Check image shape"

        c, h, w = image_shape
        self.rad_h = round(rad_offset * h)
        self.rad_w = round(rad_offset * w)
        image_shape = (c, h - 2 * self.rad_h, w - 2 * self.rad_w)
        self.init_conv(image_shape, net_params)
        if spatial_softmax:
            self.latent_dim = net_params['conv'][-1][1] * 2
        else:
            self.latent_dim = net_params['latent']

        if proprioception_shape[-1] == 0:  # no proprioception readings
            self.encoder_type = 'pixel'

        else:  # image with proprioception
            self.encoder_type = 'multi'
            self.latent_dim += proprioception_shape[0]


    def init_conv(self, image_shape, net_params):
        conv_params = net_params['conv']
        latent_dim = net_params['latent']
        channel, height, width = image_shape
        conv_params[0][0] = channel
        layers = []
        for i, (in_channel, out_channel, kernel_size, stride) in enumerate(conv_params):
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride))
            if i < len(conv_params) - 1:
                layers.append(nn.ReLU())
            width = conv_out_size(width, kernel_size, stride)
            height = conv_out_size(height, kernel_size, stride)
            # print(width, height)

        self.convs = nn.Sequential(
            *layers
        )
        
        if self.spatial_softmax:
            self.ss = SpatialSoftmax(width, height, conv_params[-1][1])
        else:
            self.fc = nn.Linear(conv_params[-1][1] * width * height, latent_dim)
        # self.ln = nn.LayerNorm(latent_dim)
        self.apply(orthogonal_weight_init)

    def forward(self, images, proprioceptions, random_rad=True, detach=False):
        if self.encoder_type == 'proprioception':
            return proprioceptions

        if self.encoder_type == 'pixel' or self.encoder_type == 'multi':
            images = images / 255.
            n, c, h, w = images.shape
            if random_rad:
                images = random_augment(images, self.rad_h, self.rad_w)
            else:                
                images = images[:, :,
                         self.rad_h: h - self.rad_h,
                         self.rad_w: w - self.rad_w,
                         ]

            if self.spatial_softmax:
                h = self.ss(self.convs(images))
            else:
                h = self.fc(self.convs(images).view((n, -1)))

            if detach:
                h = h.detach()

            if self.encoder_type == 'multi':
                h = torch.cat([h, proprioceptions], dim=-1)

            return h
        else:
            raise NotImplementedError('Invalid encoder type')


if __name__ == '__main__':
    net_params = {
        'conv': [
            # in_channel, out_channel, kernel_size, stride
            [-1, 32, 3, 2],
            [32, 64, 3, 2],
            [64, 64, 3, 2],
        ],

        'latent': 50,
    }
    image_shape = (18, 84, 84)
    proprioception_shape = (20,)
    rad_offset = 0.02
    spatial_softmax = True
    device = torch.device("cpu")

    img = torch.randn(image_shape).unsqueeze(0).to(device)
    prop = torch.randn(proprioception_shape).unsqueeze(0).to(device)

    encoder = SSEncoderModel(image_shape, proprioception_shape, net_params, rad_offset, spatial_softmax=spatial_softmax)
    phi = encoder.forward(images=img, proprioceptions=prop, random_rad=False, detach=False)
    print(phi.shape, torch.sum(phi))
