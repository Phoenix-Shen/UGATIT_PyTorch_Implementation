"""
U-GAT-IT:自适应图层实例规范化的无监督图像翻译网络。
现有的CycleGAN、UNIT、MUNIT、DRIT等受数据分布限制,
无法稳定有效地适应纹理和形状在不同程度上的变化，
U-GAT-IT通过2个设计实现了具有更强鲁棒性的端到端图像翻译模型。
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, n_features: int, bias: bool) -> None:
        super().__init__()


class Generator(nn.Module):
    """
    The generator of UGATIT which contains a downsampling block, a upsampling block and an attention module.

    Args:

    """

    def __init__(self, input_nc: int, output_nc: int, n_hiddens=64, n_resblocks=6, img_size=256, light=False) -> None:
        super().__init__()
        assert n_resblocks >= 0
        # save arguments to member varables
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_hiddens = n_hiddens
        self.n_resblocks = n_resblocks
        self.img_size = img_size
        self.light = light

        # Encoder Down sampling -- Compress image aspect
        # # [3,256,256]->[3,262,262]->[n_hiddens,256,256]
        DownBlock = list()
        DownBlock += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, n_hiddens, kernel_size=7,
                      stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(n_hiddens),
            nn.ReLU(True),  # do operation in-place
        ]

        # # [n_hiddens,256,256]  ->[n_hiddens,258,258]  ->[n_hiddens*2,128,128]
        # # [n_hiddens*2,128,128]->[n_hiddens*2,130,130]->[n_hiddens*4,64,64]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                nn.ReplicationPad2d(1),
                nn.Conv2d(n_hiddens*mult, n_hiddens*mult*2,
                          kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(n_hiddens*mult*2),
                nn.ReLU(True),
            ]

        # Encoder Bottleneck
