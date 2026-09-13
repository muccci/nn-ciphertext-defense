import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class unet_enc_128(nn.Module):
    def __init__(self, dim, nc=3):
        super(unet_enc_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim), [h5, h4, h3, h2, h1]

class unet_dec_128(nn.Module):
    def __init__(self, dim, nc=1):
        super(unet_dec_128, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.ReLU()
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*8) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*8) x 32 x 32
        self.upc5 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf*4) x 64 x 64
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Tanh() # --> [-1, 1]
                #nn.Sigmoid()
                # state size. (nc) x 128 x 128
                )

    def forward(self, vec):
        x, [h5, h4, h3, h2, h1] = vec
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, h5], 1))
        d3 = self.upc3(torch.cat([d2, h4], 1))
        d4 = self.upc4(torch.cat([d3, h3], 1))
        d5 = self.upc5(torch.cat([d4, h2], 1))
        d6 = self.upc6(torch.cat([d5, h1], 1))
        return d6

class unet_128(nn.Module):
    def __init__(self, nc, dim=128):
        super(unet_128, self).__init__()
        self.enc = unet_enc_128(nc=nc, dim=dim)
        self.dec = unet_dec_128(nc=nc, dim=dim)

    def forward(self, x):
        return self.dec(self.enc(x))     

class ae_128(nn.Module):
    def __init__(self, nc, dim=128):
        super(ae_128, self).__init__()
        self.enc = image_encoder_128(nc=nc, dim=dim)
        self.dec = image_decoder_128(nc=nc, dim=dim)

    def forward(self, x):
        return self.dec(self.enc(x))   


class image_cls_128(nn.Module):
    def __init__(self, dim, nc=3):
        super(image_cls_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.ReLU()
                )
        self.fc = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        out = h6.view(-1, self.dim)
        out = self.fc(out)
        return out


class concat_trace_encoder(nn.Module):
    def __init__(self, encoder1, encoder2, in_dim, out_dim):
        super(concat_trace_encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.concat_encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            
            nn.Linear(out_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x1, x2):
        enc1 = self.encoder1(x1)
        enc2 = self.encoder2(x2)
        out = self.concat_encoder(torch.cat([enc1, enc2], -1))
        return out


class dense_trace_encoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(dense_trace_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),

            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            
            nn.Linear(h_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class trace_encoder_32(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_32, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 32 x 32
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf*2) x 16 x 16
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*4) x 8 x 8
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*8) x 4 x 4
        self.c4 = nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4.view(-1, self.dim)

class trace_encoder_64(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_64, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf*2) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*4) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*8) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim)

class trace_encoder_128(nn.Module):
    def __init__(self, dim, nc=1):
        super(trace_encoder_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        x = F.normalize(x)
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class image_decoder_32(nn.Module):
    def __init__(self, dim, nc=1):
        super(image_decoder_32, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 4, 4, 1, 0),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU()
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*8) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 2, nf)
        # state size. (nf*4) x 16 x 16
        self.upc4 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Tanh() # --> [-1, 1]
                #nn.Sigmoid()
                # state size. (nc) x 32 x 32
                )

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        return d4

class image_encoder_32(nn.Module):
    def __init__(self, dim, nc=3):
        super(image_encoder_32, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 32 x 32
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 16 x 16
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 8 x 8
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 4 x 4
        self.c4 = nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4.view(-1, self.dim)

class image_decoder_64(nn.Module):
    def __init__(self, dim, nc=1):
        super(image_decoder_64, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.ReLU()
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Tanh() # --> [-1, 1]
                #nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        return d5

class image_decoder_96(nn.Module):
    """96x96 decoder used by DenseNet/ImageNet50-96 recovery (restored from April ckpt shapes)."""
    def __init__(self, dim, nc=1):
        super(image_decoder_96, self).__init__()
        self.dim = dim
        nf = 64
        # 1x1 -> 3x3
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, nf * 8, 3, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.ReLU()
                )
        # 3x3 -> 6x6
        self.upc2 = dcgan_upconv(nf * 8, nf * 8)
        # 6x6 -> 12x12
        self.upc3 = dcgan_upconv(nf * 8, nf * 4)
        # 12x12 -> 24x24
        self.upc4 = dcgan_upconv(nf * 4, nf * 2)
        # 24x24 -> 48x48
        self.upc5 = dcgan_upconv(nf * 2, nf)
        # 48x48 -> 96x96
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Tanh()
                )

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        d6 = self.upc6(d5)
        return d6

class image_encoder_64(nn.Module):
    def __init__(self, dim, nc=3, return_h=False):
        super(image_encoder_64, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim)


class image_encoder_96(nn.Module):
    """96x96 encoder used by DenseNet/ImageNet50-96 recovery (restored from April ckpt shapes)."""
    def __init__(self, dim, nc=3):
        super(image_encoder_96, self).__init__()
        self.dim = dim
        nf = 64
        # 96 -> 48
        self.c1 = dcgan_conv(nc, nf)
        # 48 -> 24
        self.c2 = dcgan_conv(nf, nf * 2)
        # 24 -> 12
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # 12 -> 6
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # 6 -> 3
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # 3 -> 1
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 3, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)


class image_decoder_224(nn.Module):
    """224x224 decoder for chest recon (1 -> 7 -> 14 -> 28 -> 56 -> 112 -> 224)."""

    def __init__(self, dim, nc=1):
        super(image_decoder_224, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(dim, nf * 8, 7, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(),
        )
        self.upc2 = dcgan_upconv(nf * 8, nf * 8)
        self.upc3 = dcgan_upconv(nf * 8, nf * 4)
        self.upc4 = dcgan_upconv(nf * 4, nf * 2)
        self.upc5 = dcgan_upconv(nf * 2, nf)
        self.upc6 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        return self.upc6(d5)


class image_encoder_224(nn.Module):
    """224x224 encoder for chest recon I (224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1)."""

    def __init__(self, dim, nc=1):
        super(image_encoder_224, self).__init__()
        self.dim = dim
        nf = 64
        self.c1 = dcgan_conv(nc, nf)
        self.c2 = dcgan_conv(nf, nf * 2)
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        self.c6 = nn.Sequential(
            nn.Conv2d(nf * 8, dim, 7, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh(),
        )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return self.c6(h5).view(-1, self.dim)

class image_encoder_128(nn.Module):
    def __init__(self, dim, nc=3):
        super(image_encoder_128, self).__init__()
        self.dim = dim
        nf = 64
        # state size. (nc) x 128 x 128
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, x):
        h1 = self.c1(x)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        return h6.view(-1, self.dim)

class image_decoder_128(nn.Module):
    def __init__(self, dim, nc=1):
        super(image_decoder_128, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.ReLU()
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*8) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*8) x 32 x 32
        self.upc5 = dcgan_upconv(nf * 2, nf)
        # state size. (nf*4) x 64 x 64
        self.upc6 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Tanh() # --> [-1, 1]
                #nn.Sigmoid()
                # state size. (nc) x 128 x 128
                )

    def forward(self, x):
        d1 = self.upc1(x.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        d6 = self.upc6(d5)
        return d6

class disc_128(nn.Module):
    def __init__(self, nc):
        super(disc_128, self).__init__()
        self.conv = nn.Sequential(
            dcgan_conv(nc, nf),
            dcgan_conv(nf, nf),
        )


class RefinerD128(nn.Module):
    def __init__(self, nc):
        super(RefinerD128, self).__init__()
        self.nc = nc
        self.ndf = 64
        self.main = nn.Sequential(
            # state size. (nc) x 128 x 128
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)

class RefinerD64(nn.Module):
    def __init__(self, nc):
        super(RefinerD64, self).__init__()
        self.nc = nc
        self.ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)


class RefinerD32(nn.Module):
    def __init__(self, nc):
        super(RefinerD32, self).__init__()
        self.nc = nc
        self.ndf = 64
        self.main = nn.Sequential(
            # state size. (nc) x 32 x 32
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 4 x 4
            nn.Conv2d(self.ndf * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)


class RefinerG128_BN(nn.Module):
    def __init__(self, nc, ngf):
        super(RefinerG128_BN, self).__init__()

        self.nc = nc
        self.ngf = ngf
        # 128
        self.net1 = nn.Sequential(
                    nn.Conv2d(nc, ngf, 4, 2, 1)
                    )
        self.net2 = nn.Sequential(
                    nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.net3 = nn.Sequential(
                    nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.net4 = nn.Sequential(
                    nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net5 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net6 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net7 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
                    )

        self.dnet1 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet2 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet3 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet4 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.dnet5 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 4 , ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.dnet6 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2 , ngf, 4, 2, 1),
                    nn.BatchNorm2d(ngf)
                    )
        self.dnet7 = nn.Sequential(
                    nn.ConvTranspose2d(ngf , nc, 4, 2, 1)
                    )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 128 x 128
        e1 = self.net1(input)
        # state size is (ngf) x 64 x 64
        e2 = self.net2(self.leaky_relu(e1))
        # state size is (ngf x 2) x 32 x 32
        e3 = self.net3(self.leaky_relu(e2))
        # state size is (ngf x 4) x 16 x 16
        e4 = self.net4(self.leaky_relu(e3))
        # state size is (ngf x 8) x 8 x 8
        e5 = self.net5(self.leaky_relu(e4))
        # state size is (ngf x 8) x 4 x 4
        e6 = self.net6(self.leaky_relu(e5))
        # state size is (ngf x 8) x 2 x 2
        e7 = self.net7(self.leaky_relu(e6))
        # state size is (ngf x 8) x 1 x 1
        # No batch norm on output of Encoder

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1 = self.dropout(self.dnet1(self.relu(e7)))
        # state size is (ngf x 8) x 2 x 2
        d2 = self.dropout(self.dnet2(self.relu(d1)))
        # state size is (ngf x 8) x 4 x 4
        d3 = self.dropout(self.dnet3(self.relu(d2)))
        # state size is (ngf x 8) x 8 x 8
        d4 = self.dnet4(self.relu(d3))
        # state size is (ngf x 4) x 16 x 16
        d5 = self.dnet5(self.relu(d4))
        # state size is (ngf x 2) x 32 x 32
        d6 = self.dnet6(self.relu(d5))
        # state size is (ngf) x 64 x 64
        d7 = self.dnet7(self.relu(d6))
        # state size is (nc) x 128 x 128
        output = self.tanh(d7)
        return output

class RefinerG64_BN(nn.Module):
    def __init__(self, nc, ngf):
        super(RefinerG64_BN, self).__init__()

        self.nc = nc
        self.ngf = ngf
        # 128
        self.net1 = nn.Sequential(
                    nn.Conv2d(nc, ngf, 4, 2, 1)
                    )
        self.net2 = nn.Sequential(
                    nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.net3 = nn.Sequential(
                    nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.net4 = nn.Sequential(
                    nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net5 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.net6 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    )

        self.dnet1 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet2 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet3 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.dnet4 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 4 , ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.dnet5 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2 , ngf, 4, 2, 1),
                    nn.BatchNorm2d(ngf)
                    )
        self.dnet6 = nn.Sequential(
                    nn.ConvTranspose2d(ngf , nc, 4, 2, 1)
                    )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 64 x 64
        e1 = self.net1(input)
        # state size is (ngf) x 32 x 32
        e2 = self.net2(self.leaky_relu(e1))
        # state size is (ngf x 2) x 16 x 16
        e3 = self.net3(self.leaky_relu(e2))
        # state size is (ngf x 4) x 8 x 8
        e4 = self.net4(self.leaky_relu(e3))
        # state size is (ngf x 8) x 4 x 4
        e5 = self.net5(self.leaky_relu(e4))
        # state size is (ngf x 8) x 2 x 2
        e6 = self.net6(self.leaky_relu(e5))
        # state size is (ngf x 8) x 1 x 1
        # No batch norm on output of Encoder

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1 = self.dropout(self.dnet1(self.relu(e6)))
        # state size is (ngf x 8) x 2 x 2
        d2 = self.dropout(self.dnet2(self.relu(d1)))
        # state size is (ngf x 8) x 4 x 4
        d3 = self.dropout(self.dnet3(self.relu(d2)))
        # state size is (ngf x 8) x 8 x 8
        d4 = self.dnet4(self.relu(d3))
        # state size is (ngf x 4) x 16 x 16
        d5 = self.dnet5(self.relu(d4))
        # state size is (ngf x 2) x 32 x 32
        d6 = self.dnet6(self.relu(d5))
        # state size is (ngf) x 64 x 64
        output = self.tanh(d6)
        return output


class RefinerG32_BN(nn.Module):
    def __init__(self, nc, ngf):
        super(RefinerG32_BN, self).__init__()

        self.nc = nc
        self.ngf = ngf
        # 32
        self.net1 = nn.Sequential(
                    nn.Conv2d(nc, ngf, 4, 2, 1)
                    )
        # 16
        self.net2 = nn.Sequential(
                    nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        # 8
        self.net3 = nn.Sequential(
                    nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        # 4
        self.net4 = nn.Sequential(
                    nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        # 2
        self.net5 = nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),
                    )
        # 1

        self.dnet1 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 8, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 8)
                    )
        self.dnet2 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 8 , ngf * 4, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 4)
                    )
        self.dnet3 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 4 , ngf * 2, 4, 2, 1),
                    nn.BatchNorm2d(ngf * 2)
                    )
        self.dnet4 = nn.Sequential(
                    nn.ConvTranspose2d(ngf * 2 , ngf, 4, 2, 1),
                    nn.BatchNorm2d(ngf)
                    )
        self.dnet5 = nn.Sequential(
                    nn.ConvTranspose2d(ngf , nc, 4, 2, 1)
                    )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 32 x 32
        e1 = self.net1(input)
        # state size is (ngf) x 16 x 16
        e2 = self.net2(self.leaky_relu(e1))
        # state size is (ngf x 2) x 8 x 8
        e3 = self.net3(self.leaky_relu(e2))
        # state size is (ngf x 4) x 4 x 4
        e4 = self.net4(self.leaky_relu(e3))
        # state size is (ngf x 8) x 2 x 2
        e5 = self.net5(self.leaky_relu(e4))
        # state size is (ngf x 8) x 1 x 1
        # No batch norm on output of Encoder

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1 = self.dropout(self.dnet1(self.relu(e5)))
        # state size is (ngf x 8) x 2 x 2
        d2 = self.dropout(self.dnet2(self.relu(d1)))
        # state size is (ngf x 8) x 4 x 4
        d3 = self.dropout(self.dnet3(self.relu(d2)))
        # state size is (ngf x 8) x 8 x 8
        d4 = self.dnet4(self.relu(d3))
        # state size is (ngf x 4) x 16 x 16
        d5 = self.dnet5(self.relu(d4))
        output = self.tanh(d5)
        return output


class ResDecoder128(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResDecoder128, self).__init__()
        self.dim = dim
        self.nc = nc
        self.main = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.main(input)
        return output

class ResDecoder32(nn.Module):
    def __init__(self, dim, nc, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResDecoder32, self).__init__()
        self.dim = dim
        self.nc = nc
        self.main = nn.Sequential(
            # state size. (1) x 1 x 1
            nn.ConvTranspose2d(self.dim, self.dim, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 4 x 4
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (1) x 8 x 8
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            ResnetBlock(self.dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(self.dim, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.main(input)
        return output

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out