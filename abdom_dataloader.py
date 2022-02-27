import os
from torch.utils.data import Dataset
#from PIL import Image
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# from torch.utils.tensorboard import SummaryWriter
# from utils import decode_segmap

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', 'npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]



class UnalignedDataset(Dataset):

    # A: MR dataset (source)
    # B: CT dataset (target)
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, dataroot, train = True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        # print(opt.phase)

        if train:
            self.dir_A = os.path.join(dataroot, 'train' + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(dataroot, 'train' + 'B')  # create a path '/path/to/data/trainB'
        else:
            self.dir_A = os.path.join(dataroot, 'test' + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(dataroot, 'test' + 'B')  # create a path '/path/to/data/trainB'
        # self.dir_A = os.path.join(dataroot, 'batch' + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(dataroot, 'batch' + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, float("inf")))   # load images from '/path/to/data/trainA'
        # print(self.A_paths)
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))    # load images from '/path/to/data/trainB'


        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.train = train

        # self.a_max = 3095.
        # self.a_min = -3024.

        # self.b_max = 2648.
        # self.b_min = 0.

        self.a_max = 400.
        self.a_min = -200.

        self.b_max = 1500.
        self.b_min = 50.


        # self.A_transforms = torch.nn.Sequential(transforms.RandomRotation(degrees=(0, 180)), 
        #                                         transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3)))
        # self.B_transforms = torch.nn.Sequential(transforms.RandomRotation(degrees=(0, 180)), 
        #                                         transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3)))

    def A_transforms(self, image, mask):
        angle = random.randrange(-180, 180)
        image = TF.rotate(image, angle=angle)
        mask = TF.rotate(mask, angle=angle)

        return image, mask

    def B_transforms(self, image, mask):
        angle = random.randrange(-180, 180)
        image = TF.rotate(image, angle=angle)
        mask = TF.rotate(mask, angle=angle)

        return image, mask

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # print(A_path)
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # data_A = torch.load(A_path)
        # data_B = torch.load(B_path)

        data_A = np.load(A_path).astype(float)
        data_B = np.load(B_path).astype(float)


        data_A = torch.from_numpy(data_A).unsqueeze(0).float()
        # print(data_A.size())
        data_B = torch.from_numpy(data_B).unsqueeze(0).float()

        mask_A_path = A_path.replace("A","A_labels")
        mask_A_path = mask_A_path.replace('image','label')


        mask_B_path = B_path.replace("B","B_labels")
        mask_B_path = mask_B_path.replace('image','label')


        mask_A_path = mask_A_path.replace('batch','train')
        mask_B_path = mask_B_path.replace('batch','train')


        A_mask = torch.from_numpy(np.load(mask_A_path))
        B_mask = torch.from_numpy(np.load(mask_B_path))
        # print(A_mask.size())

        data_A = torch.clip(data_A, min=self.a_min, max=self.a_max)
        data_B = torch.clip(data_B, min=self.b_min, max=self.b_max)

        A = (data_A-self.a_min)/(self.a_max-self.a_min) #.repeat(1, 5, 1, 1)

        B = (data_B-self.b_min)/(self.b_max-self.b_min) #.repeat(1, 5, 1, 1)

        if self.train:
            A, A_mask = self.A_transforms(A, A_mask)
            B, B_mask = self.B_transforms(B, B_mask)

        A = A*2-1
        B = B*2-1
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

# os.makedirs('/home/xinwen/Desktop/Stage1-toggling/trainabdom/output/data')
# writer = SummaryWriter('/home/xinwen/Desktop/Stage1-toggling/trainabdom/output/data')



# trainset = UnalignedDataset('/home/xinwen/Desktop/Stage1-toggling/abdom/npzs')
# train_loader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=8,
#     shuffle=True,
#     num_workers=1,
#     drop_last=False,
#     pin_memory=True
# )

# a_max = 0.
# a_min = 0.

# b_max = 0.
# b_min = 0.

# for i, data in enumerate(train_loader):
#     real_A = data['A']
#     real_B = data['B']
#     # self.image_paths = input['A_paths' if AtoB else 'B_paths']
#     A_mask = data['A_mask'].squeeze(1)
#     B_mask = data['B_mask'].squeeze(1)
#     # print(real_A.size(), real_B.size(), A_mask.size(), B_mask.size())
#     a_max = max(a_max, torch.max(real_A))
#     a_min = min(a_min, torch.min(real_A))
#     b_max = max(b_max, torch.max(real_B))
#     b_min = min(b_min, torch.min(real_B))
#     print(A_mask.size(), B_mask.size())
#     print(real_A.size(), real_B.size())

#     mask = torch.argmax(torch.nn.Softmax(1)(A_mask.detach()), 1)
#     maskB = torch.argmax(torch.nn.Softmax(1)(B_mask.detach()), 1)
#     # prediction = torch.argmax(nn.Softmax(1)(self.pred_fakeB.detach()), 1)
#     rows = []
#     for image_i in range(4):
#         rgb_gt = decode_segmap(mask[image_i]).cpu()
#         rgb_gtB = decode_segmap(maskB[image_i]).cpu()
#         # rgb_pred = rgb_pred = decode_segmap(prediction[image_i])
#         pic = [(real_A[image_i].repeat(3,1,1)+1)/2,
#                 rgb_gt,
#                 (real_B[image_i].repeat(3,1,1)+1)/2, # C=3, H, W
#                 rgb_gtB]
#         rows.append(torch.cat(pic, 2)) # C=3, H, W*4

#     pic = torch.cat(rows, 1)  # C=3, H*4, W*4
#     # write_step = (total_iters+1)/self.save_interval
#     writer.add_image('pred', pic, global_step=i, dataformats='CHW')

#     if i >100:
#         break

# print(a_max, a_min)
# print(b_max, b_min)
