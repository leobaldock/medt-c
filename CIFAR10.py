#!/usr/bin/env python
# coding: utf-8
import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
import torch.autograd.profiler as profiler
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from MedT_C import MedT_C
from utils import EarlyStopping, LRScheduler
from performer_pytorch import Performer

print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")
print(
    f"Using device {torch.cuda.get_device_name(torch.cuda.current_device())}")

### CONSTANTS ###
DATA_PATH = './data'
BATCH_SIZE = 64
LOADER_WORKERS = 8
### END CONSTANTS ###

parser = argparse.ArgumentParser()
parser.add_argument('--lr-schedular', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping',
                    dest='early_stopping', action='store_true')
parser.add_argument('--resnet', dest='resnet', action='store_true')
parser.add_argument('--performer', dest='performer', action='store_true')
args = vars(parser.parse_args())


train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(32, (0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                         download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=LOADER_WORKERS,
                                           pin_memory=True)

val_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                       download=True, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=LOADER_WORKERS,
                                         pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = MedT_C(
    img_dim=32,
    in_channels=3,
    patch_dim=8,
    num_classes=10,
    feature_dim=256
).cuda()

if args['resnet']:
    model = models.resnet50().cuda()

if args['performer']:
    model = Performer(
        dim=256,
        depth=1,
        heads=8,
        causal=True
    )

device = torch.device("cuda:0" if next(
    model.parameters()).is_cuda else "cpu")

# TENSORBOARD SETUP
# print("Starting TensorBoard")
# board_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=1)

# writer = SummaryWriter('runs/CIFAR10')
# dataiter = iter(board_loader)
# images, labels = dataiter.next()
# images = images.to(device)

# writer.add_graph(model, images)
# writer.close()
# print("TensorBoard Ready")

# END TENSORBOARD SETUP

# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

print(f"Model on CUDA: {next(model.parameters()).is_cuda}")

lr = 0.001
epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=lr)
scaler = GradScaler()

# strings to save plots
loss_plot_name = 'loss'
acc_plot_name = 'accuracy'
model_name = 'model'
# either initialize early stopping or learning rate scheduler
if args['lr_scheduler']:
    print('INFO: Initializing learning rate scheduler')
    lr_scheduler = LRScheduler(optimizer)
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'lrs_loss'
    acc_plot_name = 'lrs_accuracy'
    model_name = 'lrs_model'
if args['early_stopping']:
    print('INFO: Initializing early stopping')
    early_stopping = EarlyStopping()
    # change the accuracy, loss plot names and model name
    loss_plot_name = 'es_loss'
    acc_plot_name = 'es_accuracy'
    model_name = 'es_model'


def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print("Training")
    model.train()  # TODO IMPLEMENT THIS FOR MY MODEL
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(
        len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        # Run the forward pass with autocasting.
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        # Scale loss.
        scaler.scale(loss).backward()
        # Scaled optimiser step.
        scaler.step(optimizer)
        # Update scale for next iteration.
        scaler.update()

    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy


def validate(model, test_dataloader, val_dataset, criterion):
    print("Validating")
    model.eval()  # TODO IMPLEMENT THIS FOR MY MODEL
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(test_dataloader), total=int(
        len(val_dataset)/test_dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / counter
        val_accuracy = 100. * val_running_correct / total
        return val_loss, val_accuracy


def save(
        directory,
        train_accuracy,
        val_accuracy,
        acc_plot_name,
        train_loss,
        val_loss,
        loss_plot_name,
        model_name,
        show_plots=False
):
    try:
        os.mkdir(directory)
    except FileExistsError as FEE:
        print("Directory already exists")

    print('Saving loss and accuracy plots...')
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{directory}/{acc_plot_name}.png")
    if show_plots:
        plt.show()
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{directory}/{loss_plot_name}.png")
    if show_plots:
        plt.show()

    # serialize the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), f"{directory}/{model_name}.pth")


train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    # Turn on gate training after 10 epochs.
    if epoch == 10:
        print("Begin training gates...")
        for p in model.parameters():
            p.requires_grad = True
    train_epoch_loss, train_epoch_accuracy = fit(
        model, train_loader, train_set, optimizer, criterion
    )
    val_epoch_loss, val_epoch_accuracy = validate(
        model, val_loader, val_set, criterion
    )
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    if args['lr_scheduler']:
        lr_scheduler(val_epoch_loss)
    if args['early_stopping']:
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
    print(
        f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
    print(
        f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
    save(f"outputs/{time.strftime('%d-%m-%Y_%H:%M', time.gmtime(start))}",
         train_accuracy, val_accuracy, acc_plot_name, train_loss,
         val_loss, loss_plot_name, model_name, show_plots=False)

end = time.time()
print(f"Training time: {(end-start)/60:.3f} minutes")

print('TRAINING COMPLETE')
