import torch
import torch.nn as nn

from .unet_core import *

class Unet(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.seg_n_classes = args.seg_n_classes
    self.init_ch = args.init_ch

    self.encoder = nn.ModuleList(
      [
        DoubleConv(3, self.init_ch),
        Down(self.init_ch, self.init_ch*2),
        Down(self.init_ch*2, self.init_ch*4),
        Down(self.init_ch*4, self.init_ch*8),
        Down(self.init_ch*8, self.init_ch*16),
      ]
    )

    self.decoder = nn.ModuleList(
      [
        Up(self.init_ch*16, self.init_ch*8),
        Up(self.init_ch*8, self.init_ch*4),
        Up(self.init_ch*4, self.init_ch*2),
        Up(self.init_ch*2, self.init_ch),
        OutConv(self.init_ch, self.seg_n_classes)
      ]
    )

  def forward(self, x):
    x1 = self.encoder[0](x)
    x2 = self.encoder[0](x1)
    x3 = self.encoder[0](x2)
    x4 = self.encoder[0](x3)
    x5 = self.encoder[0](x4)

    x = self.decoder[0](x5, x4)
    x = self.decoder[0](x, x3)
    x = self.decoder[0](x, x2)
    x = self.decoder[0](x, x1)
    return self.decoder[4](x)
