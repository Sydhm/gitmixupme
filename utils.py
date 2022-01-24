import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import sys
import glob


# class DADataset(Dataset):
#     def __init__(self, folder_path):
#         super(DADataset, self).__init__()

#         self.files = glob.glob((folder_path + '/image/' + '*.pth'))
#         # print(self.files)

#         #self.transforms = transforms.Normalize((0), (1))
#     def __getitem__(self, index):
#         #print(index)
#         path = self.files[index]

#         image = torch.load(path).unsqueeze(0)
#         #image = self.transforms(image)
#         mask_path = path.replace("image","label")
#         mask = torch.load(mask_path).squeeze()
#         # print(mask.size())

#         #print(self.c)

#         return {'A': image, 'A_mask': mask, 'A_paths': path}

#     def __len__(self):
#         return len(self.files)


def decode_segmap(image, nc=21):
    # image = image.cpu()
    label_colors = torch.tensor([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])/128

    r = torch.zeros(image.size(), device=torch.device('cuda'))
    g = torch.zeros(image.size(), device=torch.device('cuda'))
    b = torch.zeros(image.size(), device=torch.device('cuda'))

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = torch.stack([r, g, b], axis=2).permute(2,0,1)
    # rgb = 
    return rgb

def dice_eval(pred, labels, n_class):

    compact_pred = torch.argmax(torch.nn.Softmax(1)(pred), 1)
    dice_arr = []
    dice = 0
    eps = 1e-7
    pred = torch.nn.functional.one_hot(compact_pred, num_classes = n_class)

    pred = pred.permute(0, 3, 1, 2)  # from (BHW, #class) to (B,C,HW)

    for i in range(n_class):
        inse = torch.sum(pred[:, i, :, :] * labels[:, i, :, :])
        union = torch.sum(pred[:, i, :, :]) + torch.sum(labels[:, i, :, :])
        dice = dice + 2.0 * inse / (union + eps)
        dice_arr.append(2.0 * inse / (union + eps))
    dice_arr = torch.stack(dice_arr,0)
    return dice_arr

def read_lists(fid):
    """read test file list """

    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        my_list.append(_item.split('\n')[0])
    return my_list



#https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py
class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        img1 = (img1+1)/2*225
        img2 = (img2+1)/2*225
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
