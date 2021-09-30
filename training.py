import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer():
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, pred_fn, device="cpu", ddp=False, rank=None):
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._pred_fn = pred_fn
        self._device = device

        if ddp and not rank:
            raise ValueError("DistributedDataParallel enabled, but no rank supplied to Trainer. Please set rank when initializing the Trainer.")

        self._ddp = ddp
        self._rank = rank

        self._train_loss, self._train_accuracy = [], []
        self._val_loss, self._val_accuracy = [], []
        self._model_state = []

        self.tqdm = tqdm
        self.tqdm_kwargs = {}

        self._callbacks = {}

    def train(self, epochs, quiet=False):
        for i in self.tqdm(range(epochs), disable=self._rank != 0, **self.tqdm_kwargs):
            if self._ddp:
                self._train_loader.sampler.set_epoch(i)
            if i in self._callbacks:
                callback = self._callbacks[i]["fn"]
                args = self._callbacks[i]["args"]
                kwargs = self._callbacks[i]["kwargs"]
                callback(*args, **kwargs)
            train_epoch_loss, train_epoch_accuracy = self.fit()
            val_epoch_loss, val_epoch_accuracy = self.validate()
            self._train_loss.append(train_epoch_loss)
            self._train_accuracy.append(train_epoch_accuracy)
            self._val_loss.append(val_epoch_loss)
            self._val_accuracy.append(val_epoch_accuracy)
            self._model_state.append(self._model.state_dict())

            # TODO implement this code to allow lr scheduling and early stopping.
            # if args['lr_scheduler']:
            #     lr_scheduler(val_epoch_loss)
            # if args['early_stopping']:
            #     early_stopping(val_epoch_loss)
            #     if early_stopping.early_stop:
            #         break

    def fit(self):
        # Set model to training mode.
        self._model.train()
        running_loss = 0.0
        running_correct = 0
        counter = 0
        total = 0
        batches = enumerate(self._train_loader)
        for i, data in batches:
            counter += 1
            data, target = data[0].to(self._device), data[1].to(self._device)
            total += target.size(0)
            self._optimizer.zero_grad()
            outputs = self._model(data)
            loss = self._loss_fn(outputs, target)
            running_loss += loss.item()
            preds = self._pred_fn(outputs)
            running_correct += (preds == target).sum().item()
            loss.backward()
            self._optimizer.step()

        loss = running_loss / counter
        accuracy = 100. * running_correct / total
        return loss, accuracy


    def validate(self):
        # Set model to evaluation mode.
        self._model.eval()
        running_loss = 0.0
        running_correct = 0
        counter = 0
        total = 0
        batches = enumerate(self._val_loader)
        with torch.no_grad():
            for i, data in batches:
                counter += 1
                data, target = data[0].to(self._device), data[1].to(self._device)
                total += target.size(0)
                outputs = self._model(data)
                loss = self._loss_fn(outputs, target)
                running_loss += loss.item()
                preds = self._pred_fn(outputs)
                running_correct += (preds == target).sum().item()

            val_loss = running_loss / counter
            val_accuracy = 100. * running_correct / total
            return val_loss, val_accuracy

    def save_model(self, path):
        torch.save(self._model.state_dict(), f"{path}.pth")

    def plot_loss(self, quiet=False, save_path=None):
        plt.figure(figsize=(10, 7))
        plt.plot(self._train_loss, color='orange', label='train loss')
        plt.plot(self._val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if save_path:
            plt.savefig(f"{save_path}.png")
        if not quiet:
            plt.show()
    
    def plot_accuracy(self, quiet=False, save_path=None):
        plt.figure(figsize=(10, 7))
        plt.plot(self._train_accuracy, color='orange', label='train accuracy')
        plt.plot(self._val_accuracy, color='red', label='validataion accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        if save_path:
            plt.savefig(f"{save_path}.png")
        if not quiet:
            plt.show()

    def set_callback(self, epoch, fn, *args, **kwargs):
        """
        Set fn to be called at the beginning of the specified epoch.
        """
        self._callbacks[epoch] = {"fn": fn, "args": args, "kwargs": kwargs}