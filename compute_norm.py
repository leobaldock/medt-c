import os
import argparse
import json
import numpy as np
import random
from typing import Tuple
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import nibabel as nib
from training import Trainer
from notionIntegration import tqdm_notion
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize

from MedT_C import MedT_C

load_dotenv()

METADATA_FILENAME = "meta_data_with_label.json"
    
class ADNI(Dataset):
    """ADNI Dataset."""

    def __init__(self, root, transform=None):
        self._transform = transform
        self._root = root

        # Load information from metadata file.
        metadata_file = open(os.path.join(self._root, METADATA_FILENAME))
        self._metadata = json.load(metadata_file)
        self._data = next(os.walk(os.path.join(root, "images")), (None, None, []))[2]  # [] if no file

    def __len__(self):
        return len(self._metadata)*16 # Because we took 16 slices.

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        filename = self._data[idx]
        nifti_image = nib.load(os.path.join(self._root, "images", filename))
        nii_data = nifti_image.get_fdata()
        # image = Image.fromarray(nii_data)
        image = nii_data

        if self._transform:
            image = self._transform(image)

        label = self._metadata[filename.split("_")[0]]["label"]

        return (image, label)

    # class Resize(object):
    #     def __init__(self, output_size):
    #         assert isinstance(output_size, (int, tuple))
    #         if isinstance(output_size, int):
    #             self.output_size = (output_size, output_size)
    #         else:
    #             assert len(output_size) == 2
    #             self.output_size = output_size
        
    #     def __call__(self, sample):
    #         image = sample

    #         h, w = image.shape[:2]
    #         new_h, new_w = self.output_size

    #         top = np.random.randint(0, h - new_h)
    #         left = np.random.randint(0, w - new_w)

    #         image = image[top: top + new_h,
    #                     left: left + new_w]

    #         landmarks = landmarks - [left, top]

    #         return {'image': image, 'landmarks': landmarks}


transform = transforms.Compose(
    [
        # transforms.Resize((224,224)),
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

dataset = ADNI("./data/ADNI_sliced", transform=transform)

img = dataset.__getitem__(50)[0]
plt.imshow(img)

# image_loader = DataLoader(dataset = dataset, 
#                           batch_size = 64, 
#                           shuffle = False, 
#                           num_workers = 8,
#                           pin_memory = True)

# # placeholders
# psum = torch.tensor([0.0])
# psum_sq = torch.tensor([0.0])

# # loop through images
# for inputs in tqdm(image_loader):
#     print(inputs.shape)
#     break
#     psum += inputs.sum(axis = [0, 2, 3])
#     psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

# (images, labels) = next(iter(image_loader))
# def show(img):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     plt.show()

# show(make_grid(images.cpu().data))