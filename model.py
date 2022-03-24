"""
U-GAT-IT:自适应图层实例规范化的无监督图像翻译网络。
现有的CycleGAN、UNIT、MUNIT、DRIT等受数据分布限制,
无法稳定有效地适应纹理和形状在不同程度上的变化，
U-GAT-IT通过2个设计实现了具有更强鲁棒性的端到端图像翻译模型。
"""
# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import weight_init


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

    def __init__(self, n_features: int, activation="relu") -> None:
        super().__init__()
        self.activation = activation
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
            0].unsqueeze(2).unsqueeze(3)
        gmp_weight = list(self.gmp_fc.parameters())[
            0].unsqueeze(2).unsqueeze(3)
        # multiply with the input data, we get the attention map
        gap = x*gap_weight
        gmp = x*gmp_weight
        # [1,n_features,H,W]->[1,2*n_features,H,W]
        x = t.cat([gap, gmp], 1)
        # Dimensionality reduction
        # [1,2*n_features,H,W]->[1,n_features,H,W]
        if self.activation == "relu":
            x = F.relu(self.conv1x1(x), True)
        elif self.activation == "leaky_relu":
            x = F.leaky_relu(self.conv1x1(x), 0.2, True)
        else:
            raise NotImplementedError(
                "only support relu and leaky_relu acitvation function")
        # get the attention map via sum operation
        # [1,1,H,W]
        heatmap = t.sum(x, dim=1, keepdim=True)
        return x, cam_logit, heatmap


class BetaAndGamma(nn.Module):
    """
    Extracting learnable parameters beta and gamma from feature maps in the CAM module

    Args:
        n_features: the number of features in the input feature map
        light: use light mode to save memory
        feature_size: the size of extacted feature map
    """

    def __init__(self, n_features: int, feature_size: int, light: bool) -> None:
        super().__init__()
        # if use light mode, we do the pooling operation first
        # [n_features,H,W]->[1,n_features]
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
        # [n_features,H,W]->[1,n_features]
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
        """
        args: 
            x: the input feature map

        returns:
            tuple of gamma and beta
        """
        x = self.fc(x)
        beta = self.beta(x)
        gamma = self.gamma(x)
        return gamma, beta


class ResnetAdaILNBlock(nn.Module):
    """
    Resnet block with Adaptive Layer Instance Normlization
    """

    def __init__(self, n_features: int, bias: bool) -> None:
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(n_features, n_features,
                               kernel_size=3, stride=1, padding=0, bias=bias)
        self.norm1 = AdaILN(n_features)
        self.relu1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(n_features, n_features,
                               kernel_size=3, stride=1, padding=0, bias=bias)
        self.norm2 = AdaILN(n_features)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return x+out


class AdaILN(nn.Module):
    """
    Adaptive Instance Layer Normalization in the Paper

    Args:
        n_features: the input features
        eps: in the computation progress of normlization, Adding eps is to prevent the divisor from being 0
    """

    def __init__(self, n_features: int, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(t.ones((1, n_features, 1, 1)))
        self.rho.data.fill_(0.9)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # IN normalizes the 2nd and 3rd dimensions of the image
        in_mean, in_var = t.mean(x, dim=[2, 3], keepdim=True), t.var(
            x, dim=[2, 3], keepdim=True)
        # LN normalizes the 1st, 2nd and 3rd dimensions of the image
        ln_mean, ln_var = t.mean(x, dim=[1, 2, 3], keepdim=True), t.var(
            x, dim=[1, 2, 3], keepdim=True)
        # compute the Normalized value, Adding eps is to prevent the divisor from being 0
        in_value = (x-in_mean)/t.sqrt(in_var+self.eps)
        ln_value = (x-ln_mean)/t.sqrt(ln_var+self.eps)
        # Adaptivly combine the in_value and the ln_value
        # Copy parameters in the case of multiple batches
        rho = self.rho.expand(x.shape[0], -1, -1, -1)
        out = rho*in_value + (1-rho)*ln_value
        # multiply with GAMMA and BETA (reshape operation is needed, also we can unsqueeze them)
        out = out * gamma.unsqueeze(2).unsqueeze(3) + \
            beta.unsqueeze(2).unsqueeze(3)
        return out


class ILN(nn.Module):
    """
    Common Instance Layer Normalization Layer\n
    The Module Contains 3 Parameters : GAMMA BETA and RHO, thay are all learnable

    Args:
        n_features: the input features
        eps: in the computation progress of normlization, Adding eps is to prevent the divisor from being 0

    """

    def __init__(self, n_features: int, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(t.zeros((1, n_features, 1, 1)))
        self.gamma = nn.Parameter(t.ones((1, n_features, 1, 1)))
        self.beta = nn.Parameter(t.zeros((1, n_features, 1, 1)))

    def forward(self, x: Tensor) -> Tensor:
        # IN normalizes the 2nd and 3rd dimensions of the image
        in_mean, in_var = t.mean(x, dim=[2, 3], keepdim=True), t.var(
            x, dim=[2, 3], keepdim=True)
        # LN normalizes the 1st, 2nd and 3rd dimensions of the image
        ln_mean, ln_var = t.mean(x, dim=[1, 2, 3], keepdim=True), t.var(
            x, dim=[1, 2, 3], keepdim=True)
        # compute the Normalized value, Adding eps is to prevent the divisor from being 0
        in_value = (x-in_mean)/t.sqrt(in_var+self.eps)
        ln_value = (x-ln_mean)/t.sqrt(ln_var+self.eps)
        # Adaptivly combine the in_value and the ln_value
        # Copy parameters in the case of multiple batches
        rho = self.rho.expand(x.shape[0], -1, -1, -1)
        out = rho*in_value + (1-rho)*ln_value
        # multiply with GAMMA and BETA (Need to extend dimensions to cope with multiple batches)
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1) + \
            self.beta.expand(x.shape[0], -1, -1, -1)
        return out


class Generator(nn.Module):
    """
    The generator of UGATIT which contains a downsampling block, a upsampling block and an attention module.

    Args: 
        input_nc: the number of channels in the input images
        output_nc: the number of channels in the output images
        n_hiddens: the number of features in the hidden layers
        n_resblocks: the number of the residual blocks in the down and up sampling procedure
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

        self.DownBlock = nn.Sequential(*DownBlock)
        # Class Activation Map
        # [n_hiddens*4,64,64]->([n_hiddens*4,64,64],[1,2],[1,64,64])
        self.cam = ClassActivationMapping(n_hiddens*mult)

        # Gamma, Beta Blocks
        # [n_hiddens*4,64,64]->[1,n_hiddens*4]
        self.beta_gamma = BetaAndGamma(
            n_hiddens*mult, self.img_size // mult, light=self.light)

        # Up Sampling Bottleneck
        for i in range(n_resblocks):
            setattr(self, "UpBlock1_"+str(i+1),
                    ResnetAdaILNBlock(mult*n_hiddens, bias=False))

        # Up Sampling Operation
        # # [n_hiddens*4,64,64]->[n_hiddens*4,128,128]->[n_hiddens*2,128,128]
        # # [n_hiddens*2,128,128]->[n_hiddens*2,256,256]->[n_hiddens,256,256]
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            UpBlock2 += [
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_hiddens*mult, int(n_hiddens*mult/2),
                          kernel_size=3, stride=1, padding=0, bias=False),
                ILN(int(n_hiddens*mult/2)),
                nn.ReLU(True),
            ]
        # # [n_hiddens,256,256]->[n_hiddens,262,262]->[output_nc,256,256]->[Tanh-Rescale to [0-1]]
        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(n_hiddens, output_nc, kernel_size=7,
                               stride=1, padding=0, bias=False),
                     nn.Tanh(),
                     ]

        self.UpBlock2 = nn.Sequential(*UpBlock2)
        self.apply(weight_init)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        input: Tensor of 4 dim \n
        output: Generated Image (B*C*H*W), log Probability of the Image (B*2), HeatMap (B*1*64*64)
        """
        # DownSampling Operation
        x = self.DownBlock.forward(x)
        # Cam Module
        x, cam_logit, attention = self.cam.forward(x)
        # Gamma and Beta
        gamma, beta = self.beta_gamma.forward(x)
        # UpSampling ResBlock with AdaILN Module
        for i in range(self.n_resblocks):
            x = getattr(self, "UpBlock1_"+str(i+1)).forward(x, gamma, beta)
        out = self.UpBlock2.forward(x)
        # return the generated image, the cam log probability and the attention heatmap
        return out, cam_logit, attention


class Discriminator(nn.Module):
    """
    The structure of the discriminator is similar to that of the generator,
    consisting of a downsampling module and a CAM module

    Args:
        input_ch: the channel of input images
        n_hiddens: the features of the hidden layer
        n_layers: the number of down sampling layer
    """

    def __init__(self, input_ch: int, n_hiddens=64, n_layers=5) -> None:
        super().__init__()
        # Down Sampling
        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_ch, n_hiddens, kernel_size=4,
                      stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(1, n_layers-2):
            mult = 2**(i-1)
            model += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(n_hiddens*mult, n_hiddens*mult*2,
                          kernel_size=4, stride=2, padding=0, bias=True),
                nn.LeakyReLU(0.2, True)
            ]
        mult = 2**(n_layers-2-1)
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_hiddens*mult, n_hiddens*mult*2,
                      kernel_size=4, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
        ]
        self.model = nn.Sequential(*model)

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.cam = ClassActivationMapping(
            n_hiddens*mult, activation="leaky_relu")
        # Fianl Conv layer
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(n_hiddens*mult, 1, kernel_size=4,
                              stride=1, padding=0, bias=False)

        self.apply(weight_init)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward function of the Dicriminator

        Args:
            x: the input feature map
        Returns:
            A tuple that conbines the output,the cam_logit and the heatmap 
        """
        # down Sampling
        x = self.model.forward(x)
        # Class Activation Mapping
        x, cam_logit, heatmap = self.cam.forward(x)
        # Padding and get the result
        x = self.pad(x)
        x = self.conv(x)
        return x, cam_logit, heatmap


class RhoClipper(object):
    def __init__(self, min, max) -> None:
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module: nn.Module) -> None:
        if hasattr(module, "rho"):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
