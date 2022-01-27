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
# class MyLoss(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input_):
#         ctx.save_for_backward(input_)
#         output = input_
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         # _, alpha_ = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output/64 
#         return grad_input

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
            # nn.Conv2d(128, 256, kernel_size=4, stride=2), norm_layer(256, affine=True, eps=eps), ## output 2*2
            # Rearrange('b c (h) (w) -> b c (h w)'),
        )

    def forward(self, x):
        # print(x.size())
        x = self.sequence(x)#.mean(-1)
        # x = self.sigmoid(x)
        return x

class Domain_classifier(nn.Module):
    def __init__(self, ad_mode='universal'):
        super(Domain_classifier, self).__init__()
        eps = 0.001
        norm_layer=nn.InstanceNorm2d
        if ad_mode == 'universal':
            self.ad_net = AdversarialNetwork(1)
        elif ad_mode == 'local':
            self.ad_net = nn.ModuleList([AdversarialNetwork(emb_size) for i in range(64)])
        else:
            raise ValueError("ad_mode undifined")
        self.ad_mode = ad_mode
        # self.resnetblock = nn.Sequential(nn.Conv2d(128, 128, 3, padding = 1), nn.InstanceNorm2d(128, affine=True, eps=eps), nn.LeakyReLU(0.2, True),
        #                                         nn.Conv2d(128, 128, 3, padding = 1), nn.InstanceNorm2d(128, affine=True, eps=eps),)
        # self.linear = nn.Sequential(Rearrange('b c l -> b l c'),   # b, 256, 64 -> b, 64, 256
        #                             nn.Linear(256, 128), nn.LeakyReLU(), nn.Dropout(0.5),
        #                             nn.Linear(128, 1),
        #                             Rearrange('b (h w) c -> b c h w', h=8))  # b, 64, 1 
        # self.classificationhead = nn.Conv2d(256, 1, 1)
        self.classificationhead = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2), norm_layer(256, affine=True, eps=eps), nn.LeakyReLU(0.2, True), 
                                                nn.Conv2d(256, 512, kernel_size=4, stride=2), norm_layer(512, affine=True, eps=eps), nn.LeakyReLU(0.2, True),
                                                nn.Conv2d(512, 1, kernel_size=3, stride=1))
    def forward(self, x):

        ## crop out patches
        b, c, _, _ = x.size()
        x = x.view(b,c,32,-1)

        pred = []

        for patch_id in range(64):
            patch = x[:,:,:,32*patch_id:32*(patch_id+1)]

            if self.ad_mode == 'universal':
                pred += [self.ad_net(patch)]
                # print(self.ad_net(patch).size())
            else:
                pred += [self.ad_net[patch_id](patch)]
        
        output = torch.cat(pred, -1).view(b,128,48,48)
        # output = torch.stack(pred, -1).view(b,256,8,8)
        # print(output.size())
        # print(pred[0].size())
        # output = output + self.resnetblock(output)
        # output = self.linear(output)
        # output = torch.stack(pred).view(b,128,8,8)

        # output = self.ad_net(x)

        output = self.classificationhead(output)
        # output = torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest')

        return output