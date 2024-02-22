# https://arxiv.org/pdf/2204.01697.pdf
# https://github.com/google-research/maxvit

from torch import nn, Tensor


class conv_norm_act(in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.GELU(),
    )


class SqueezeExcitation(nn.Sequential):
    pass


# pre-norm MBConv
class MBConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        hidden_dim = in_dim * 4
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            conv_norm_act(in_dim, hidden_dim, 1),
            conv_norm_act(hidden_dim, hidden_dim, 3, stride, hidden_dim),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )
        
        if stride > 1:
            self.skip = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(in_dim, out_dim, 1),
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.residual(x)


# (*, H/sizeh, sizeh, W, C)
# (*, H/sizeh, sizeh, W/sizew, sizew, C)
# (*, H/sizeh, W/sizew, sizeh, sizew, C)
def block(x: Tensor, size: int) -> Tensor:
    return x.unflatten(-3, (-1, size)).unflatten(-2, (-1, size)).transpose(-3, -4)  


def unblock(x: Tensor) -> Tensor:
    return x.transpose(-3, -4).flatten(-5, -4).flatten(-3, -2)


# (*, sizeh, H/sizeh, W, C)
# (*, sizeh, H/sizeh, sizew, W/sizew, C)
# (*, H/sizeh, W/sizew, sizeh, sizew, C)
def grid(x: Tensor, size: int) -> Tensor:
    return x.unflatten(-3, (size, -1)).unflatten(-2, (size, -1)).transpose()
