import os
import argparse
import json
import numpy as np
from typing import Tuple
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import datasets, transforms
import nibabel as nib
from training import Trainer
from notionIntegration import tqdm_notion
from dotenv import load_dotenv

from MedT_C import MedT_C

########################
# Constants
########################
IMAGE_DIMENSION = 256
NUM_CLASSES = 3
IMAGE_CHANNELS = 1
METADATA_FILENAME = "meta_data_with_label.json"
########################


########################
# Hyperparameters
########################
batch_size = 64
epochs = 100
feature_dimension = 256
learning_rate = 1e-3
########################

########################
# Model Definition
########################
def create_model():
    
    model = MedT_C(IMAGE_DIMENSION, IMAGE_DIMENSION/4, NUM_CLASSES, in_channels=IMAGE_CHANNELS, feature_dim=feature_dimension)

    # Dummy forward pass to initialise weights before distributing.
    model(torch.zeros(batch_size, IMAGE_CHANNELS, IMAGE_DIMENSION, IMAGE_DIMENSION))

    return model
########################

########################
# Dataset Setup
########################
def create_datasets(data_path: str):
    class ADNI(Dataset):
        """ADNI Dataset."""

        def __init__(self, root, transform=None):
            self._transform = transform
            self._root = root

            # Load information from metadata file.
            metadata_file = open(os.path.join(data_path, METADATA_FILENAME))
            self._data = json.load(metadata_file).values() # Just discard the keys, we don't need them.

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.toList()

            path = self._data[idx]['masked']
            nifti_image = nib.load(os.path.join(data_path, path))
            nii_data = nifti_image.get_fdata()

            # Rotate for ease of viewing.
            nii_data = np.rot90(nii_data, 1, (1, 2))

            # Take 20 central slices for our purposes.
            midpoint = nii_data.shape[0] // 2
            slices_data = nii_data[midpoint - 10 : midpoint + 10, :, :]
            
            if self._transform:
                slices_data = self._transform(slices_data)

            return (slices_data, self._data[idx]['label'])


def create_data_loaders(rank: int, world_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # First make transforms for training and validation sets.
    train_transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(IMAGE_DIMENSION, (0.85, 1.15)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # Then setup the datasets with the relevant transforms.
    training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
    validation_data = datasets.CIFAR10(root="data", train=False, download=True, transform=val_transform)

    # Get the dataset variance so we can normalise the loss function.
    # data_variance = np.var(training_data.data / 255.0)

    # Create a sampler for distributed loading.
    sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True, seed=42)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, val_loader
########################

def main(rank: int, epochs: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

########################
# Train Model
########################
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    def loss_fn(output, target):
        return criterion(output, target)
    def pred_fn(output):
        _, preds = torch.max(output[1].data, 1)
        return preds
    def activate_gate_training(trainer):
        for p in trainer._model.parameters():
            p.requires_grad = True
    trainer = Trainer(model, train_loader, val_loader, optimizer, loss_fn, pred_fn, device=device, ddp=True)
    trainer.set_callback(10, activate_gate_training, trainer)
    trainer.tqdm = tqdm_notion
    trainer.tqdm_kwargs = {"page_title": "MedT-C"}
    trainer.train(epochs, quiet=args.quiet)
########################

    return trainer

if __name__ == '__main__':
########################
# Args and Env Vars
########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', default=False, action="store_true", help="Surpress script output for headless environments. Default=False.")
    args = parser.parse_args()
    load_dotenv()
########################

    rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv('WORLD_SIZE'))

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    data_path = os.getenv("ADNI_RDM_PATH")
    train_set, val_set = create_datasets(data_path)
    train_loader, val_loader = create_data_loaders(rank, world_size, batch_size)
    trainer = main(rank=rank,
                 epochs=epochs,
                 model=create_model(),
                 train_loader=train_loader,
                 val_loader=val_loader)

########################
# Save Results
########################
    if rank == 0:
        DIRECTORY = "graphs"
        try:
            os.mkdir(DIRECTORY)
        except FileExistsError as FEE:
            print(f"Directory {DIRECTORY} already exists.")
        trainer.plot_accuracy(quiet=args.quiet, save_path=os.path.join(DIRECTORY, "acc"))
        trainer.plot_loss(quiet=args.quiet, save_path=os.path.join(DIRECTORY, "loss"))
########################