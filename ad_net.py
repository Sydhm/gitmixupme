import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

x = torch.randn(8, 1, 256, 256)

# patch_size = 16 # 16 pixels
# pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        eps = 0.001
        norm_layer=nn.InstanceNorm2d
        self.sequence = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            # nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # Rearrange('b e (h) (w) -> b (h w) e'),
            nn.Conv2d(in_feature, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.2, True), # 15, 15
            nn.Conv2d(64, 128, kernel_size=4, stride=2), norm_layer(128, affine=True, eps=eps), nn.LeakyReLU(0.2, True),  #6 * 6
            nn.Conv2d(128, 128, kernel_size=4, stride=2), norm_layer(128, affine=True, eps=eps), ## output 2*2
            Rearrange('b c (h) (w) -> b c (h w)'),
        )

    def forward(self, x):
        # print(x.size())
        x = self.sequence(x).mean(-1)
        # x = self.sigmoid(x)
        return x

class Domain_classifier(nn.Module):
    def __init__(self, ad_mode='universal'):
        super(Domain_classifier, self).__init__()
        eps = 0.001
        if ad_mode == 'universal':
            self.ad_net = AdversarialNetwork(1)
        elif ad_mode == 'local':
            self.ad_net = nn.ModuleList([AdversarialNetwork(emb_size) for i in range(64)])
        else:
            raise ValueError("ad_mode undifined")
        self.ad_mode = ad_mode
        self.resnetblock = nn.Sequential(nn.Conv2d(128, 128, 3, padding = 1), nn.InstanceNorm2d(128, affine=True, eps=eps), nn.LeakyReLU(0.2, True),
                                                nn.Conv2d(128, 128, 3, padding = 1), nn.InstanceNorm2d(128, affine=True, eps=eps),)
        self.classificationhead = (nn.Conv2d(128, 1, 1, bias=False))
    def forward(self, x):

        ## crop out patches
        b, c, _, _ = x.size()
        x = x.view(b,c,32,-1)

        pred = []

        for patch_id in range(64):
            patch = x[:,:,:,32*patch_id:32*(patch_id+1)]

            if self.ad_mode == 'universal':
                pred += self.ad_net(patch)
            else:
                pred += self.ad_net[patch_id](patch)
        
        output = torch.stack(pred).view(b,128,8,8)
        output = output + self.resnetblock(output)
        output = self.classificationhead(output)

        return output