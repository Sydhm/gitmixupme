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

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 16, emb_size: int = 768, img_size: int = 256):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
            # nn.Conv2d(in_channels, 32, kernel_size=1, stride=1),
            # nn.Conv2d(32, 32, kernel_size=4, stride=4),
            # nn.Conv2d(32, 64, kernel_size=4, stride=4),
            # nn.Conv2d(64, 512, kernel_size=2, stride=2),  ## output 8*8 patch
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # print(cls_tokens.size())
        # prepend the cls token to the input
        # x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        patches = x
        # print(x.size())
        x += self.positions
        return x#, patches

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0, vis=False):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.vis = vis
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        # print(x.size())
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print(keys.size())
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        # print("max", torch.max(att))
        # att = att
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        if self.vis:
            return out, att
        else:
            return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self,
#                  emb_size: int = 768,
#                  drop_p: float = 0.,
#                  forward_expansion: int = 4,
#                  forward_drop_p: float = 0.,
#                  ** kwargs):
#         super().__init__(
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 MultiHeadAttention(emb_size, **kwargs),
#                 nn.Dropout(drop_p)
#             )),
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 FeedForwardBlock(
#                     emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
#                 nn.Dropout(drop_p)
#             )
#             ))

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 vis = False,
                 ** kwargs):
        super().__init__()
        self.vis = vis
        self.first_half = nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, vis= self.vis, **kwargs))
        self.dropout = nn.Dropout(drop_p)
        self.second_half = ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)))
    def forward(self, input):
        if self.vis:
            x, self.attn = self.first_half(input)
        else:
            x = self.first_half(input)
        x = self.dropout(x)
        x = x+input
        x = self.second_half(x)
        return x


# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth: int = 12, **kwargs):
#         super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__()
        self.sequence = nn.Sequential(*[TransformerEncoderBlock(**kwargs) for _ in range(depth-1)])
        self.last_layer = TransformerEncoderBlock(vis = True, **kwargs)
        # super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
    def forward(self, x):
        # print(x.size())
        x = self.sequence(x)
        x = self.last_layer(x)
        self.attn = self.last_layer.attn
        # print(attn.size())
        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 2):
        super().__init__(
            # Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, 256),  ## hidden layer
            nn.Linear(256, 1),  ## hidden layer
            Rearrange('b (h w) c -> b c h w', h = 8)
        )

# class ViT(nn.Sequential):
#     def __init__(self,     
#                 in_channels: int = 1,
#                 patch_size: int = 16,
#                 emb_size: int = 768,
#                 img_size: int = 256,
#                 depth: int = 12,
#                 n_classes: int = 2,
#                 **kwargs):
#         super().__init__(
#             PatchEmbedding(in_channels, patch_size, emb_size, img_size),
#             TransformerEncoder(depth, emb_size=emb_size, **kwargs),
#             ClassificationHead(emb_size, n_classes)
#         )

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        # x = self.sigmoid(x)
        return x

class ViT(nn.Module):
    def __init__(self,     
                in_channels: int = 1,
                patch_size: int = 16,
                emb_size: int = 512,
                img_size: int = 256,
                depth: int = 6,
                n_classes: int = 1,
                **kwargs):
        super().__init__()
        self.patchembedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)

        # self.ad_list = nn.ModuleList([AdversarialNetwork(emb_size) for i in range(64)]) ##### v2

        # self.ad_list = []
        # for ad_num in range(64):
        #     self.ad_list.append(AdversarialNetwork(emb_size).cuda())
        self.transformer = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)
    def forward(self, x):
        # x, patches = self.patchembedding(x)  ### x: 8*8 patch
        # b,c,_=x.size()

        x = self.patchembedding(x)
        x = self.transformer(x)
        x = self.classifier(x)
        x = torch.nn.AvgPool2d(kernel_size=2, stride=2)  # 8*8

        return x

        # local_output_list=[]

        # for ad_num, ad_net in enumerate(self.ad_list):           ########## v2
        #     local_output = ad_net(x[:,ad_num,:])  ## b,c,1

        #     local_output_list.append(local_output)

        # local_output = torch.cat(local_output_list, dim=1).view(b,1,8,8)   #b,c,64

        # return local_output

# patches_embedded = PatchEmbedding()(x)
# print(TransformerEncoderBlock()(patches_embedded).shape)
# print(ViT()(x).shape)