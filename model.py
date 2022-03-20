"""
U-GAT-IT:自适应图层实例规范化的无监督图像翻译网络。
现有的CycleGAN、UNIT、MUNIT、DRIT等受数据分布限制,
无法稳定有效地适应纹理和形状在不同程度上的变化，
U-GAT-IT通过2个设计实现了具有更强鲁棒性的端到端图像翻译模型。
"""
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResnetBlock(nn.Module):
    """
    resnet block, The passing image with constant length and width
    """

    def __init__(self, n_features: int, bias: bool) -> None:
        super().__init__()
        conv_block = list()
        conv_block += [
            nn.ReplicationPad2d(1),
            nn.Conv2d(n_features, n_features, kernel_size=3,
                      stride=1, padding=0, bias=bias),
            nn.InstanceNorm2d(n_features),
            nn.ReLU(True),

            nn.ReplicationPad2d(1),
            nn.Conv2d(n_features, n_features, kernel_size=3,
                      stride=1, padding=0, bias=bias),
            nn.InstanceNorm2d(n_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: Tensor) -> Tensor:
        out = x+self.conv_block.forward(x)
        return out


class ClassActivationMapping(nn.Module):
    """
    The attention Module for classification
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()

        self.gap_fc = nn.Linear(n_features, 1, bias=False)
        self.gmp_fc = nn.Linear(n_features, 1, bias=False)
        self.conv1x1 = nn.Conv2d(
            2*n_features, n_features, kernel_size=1, stride=1, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        the forward function

        Args: 
            input :tensor,the inpute feature map\n

        returns: 
            the feature map, the log_probability of the feature map and the attention map
        """
        # global avg and map pooling function
        # [1,n_features,H,W]->[1,n_features,1,1]
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        # [1,n_features,1,1]->[1,n_features]->[1,1]
        gap_logit = self.gap_fc.forward(gap.view(x.shape[0], -1))
        gmp_logit = self.gmp_fc.forward(gmp.view(x.shape[0], -1))
        # 2*[1,1]->[1,2]
        cam_logit = t.cat([gap_logit, gmp_logit], 1)
        # extract weights of the liear layer
        # gap_fc.weight.shape = [1,n_features]
        gap_weight = list(self.gap_fc.parameters())[
            0].reshape((1, x.shape[1], 1, 1))
        gmp_weight = list(self.gmp_fc.parameters())[
            0].reshape((1, x.shape[1], 1, 1))
        # multiply with the input data, we get the attention map
        gap = x*gap_weight
        gmp = x*gmp_weight
        # [1,n_features,H,W]->[1,2*n_features,H,W]
        x = t.cat([gap, gmp], 1)
        # Dimensionality reduction
        # [1,2*n_features,H,W]->[1,n_features,H,W]
        x = F.relu(self.conv1x1(x), True)

        # get the attention map via sum operation
        heatmap = t.sum(x, dim=1, keepdim=True)
        return x, cam_logit, heatmap


class BetaAndGamma(nn.Module):
    """
    Extracting learnable parameters beta and gamma from feature maps in the AdaLIN module

    Args:
        n_features: the number of features in the input feature map
        light: use light mode to save memory
        feature_size: the size of extacted feature map
    """

    def __init__(self, n_features: int, feature_size: int, light: bool) -> None:
        super().__init__()
        # if use light mode, we do the pooling operation first
        if light:
            fully_connection_layers = [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_features, n_features, bias=False),
                nn.ReLU(True),
                nn.Linear(n_features, n_features, bias=False),
                nn.ReLU(True),
            ]
        # else, do not execute the pooling operation
        else:
            fully_connection_layers = [
                nn.Flatten(),
                nn.Linear(feature_size*feature_size *
                          n_features, n_features, False),
                nn.ReLU(True),
                nn.Linear(n_features, n_features, bias=False),
                nn.ReLU(True),
            ]

        self.gamma = nn.Linear(n_features, n_features, bias=False)
        self.beta = nn.Linear(n_features, n_features, bias=False)
        self.fc = nn.Sequential(*fully_connection_layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.fc(x)
        beta = self.beta(x)
        gamma = self.gamma(x)
        return gamma, beta


class Generator(nn.Module):
    """
    The generator of UGATIT which contains a downsampling block, a upsampling block and an attention module.

    Args: 
        input_nc: the number of channels in the input images
        output_nc: the number of channels in the output images
        n_hiddens: the number of features in the hidden layers
        n_resblocks: the number of the residual blocks in the network
        light: use light model to save cuda memory
        img_size: the shape of the image (ususally 256)
    """

    def __init__(self, input_nc: int, output_nc: int, n_hiddens=64, n_resblocks=6, img_size=256, light=False) -> None:
        super().__init__()
        assert n_resblocks >= 0
        # save arguments to member varables
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_hiddens = n_hiddens
        self.n_resblocks = n_resblocks
        self.light = light
        self.img_size = img_size
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
        # # [n_hiddens*4,64,64]->[n_hiddens*4,64,64]
        mult = 2**n_downsampling
        for i in range(n_resblocks):
            DownBlock += [ResnetBlock(n_hiddens*mult, bias=False)]

        # Class Activation Map
        self.cam = ClassActivationMapping(n_hiddens*mult)

        # Gamma, Beta Blocks
