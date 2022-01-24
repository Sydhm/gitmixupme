import os
from torch.utils.data import Dataset
#from PIL import Image
import random
import numpy as np
import torch
from PIL import Image

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

    def __init__(self, dataroot):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        # print(opt.phase)


        self.dir_A = os.path.join(dataroot, 'train' + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(dataroot, 'train' + 'B')  # create a path '/path/to/data/trainB'
        # self.dir_A = os.path.join(dataroot, 'batch' + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(dataroot, 'batch' + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, float("inf")))   # load images from '/path/to/data/trainA'
        # print(self.A_paths)
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))    # load images from '/path/to/data/trainB'


        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

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
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # data_A = torch.load(A_path)
        # data_B = torch.load(B_path)
        data_A = torch.from_numpy(np.load(A_path)).permute(2,0,1)
        # print(data_A.size())
        data_B = torch.from_numpy(np.load(B_path)).permute(2,0,1)

        mask_A_path = A_path.replace("A","A_labels")
        mask_A_path = mask_A_path.replace('image','label')


        mask_B_path = B_path.replace("B","B_labels")
        mask_B_path = mask_B_path.replace('image','label')


        mask_A_path = mask_A_path.replace('batch','train')
        mask_B_path = mask_B_path.replace('batch','train')


        A_mask = torch.from_numpy(np.load(mask_A_path)).permute(2,0,1)
        B_mask = torch.from_numpy(np.load(mask_B_path)).permute(2,0,1)
        # print(A_mask.size())


        A = data_A #.repeat(1, 5, 1, 1)


        B = data_B #.repeat(1, 5, 1, 1)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


