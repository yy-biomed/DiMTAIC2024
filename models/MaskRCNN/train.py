import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import random

from loader import MaskDataset
from model import get_maskrcnn_resnet50_fpn_v2


####
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    loss_all = []
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_all.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print('loss: %.7f  [%.4d/%.4d]' % (loss, current, size))

    return loss_all


####
def valid(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    n_batch = len(dataloader)
    model.eval()
    valid_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            valid_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    valid_loss /= n_batch
    correct /= size
    print('###################################################################')
    print('Valid Error: \n Accuracy: %.4f, Avg loss: %.4f\n' % (100 * correct, valid_loss))

    return valid_loss, correct


if __name__ == '__main__':

    ############################################################################
    train_dir = '/tcdata/train/'
    log_dir = 'log'
    pretrained_path = 'maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'

    n_class = 3

    n_epoch = (10, 10)
    batch_size = 64
    n_worker = 4
    ############################################################################

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path('%s/00' % log_dir).mkdir(parents=True, exist_ok=True)
    Path('%s/01' % log_dir).mkdir(parents=True, exist_ok=True)

    using_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tp_label = pd.read_csv('%s/label.csv' % train_dir)
    basename = tp_label.iloc[:, 0].values.tolist()
    random.shuffle(basename)
    train_cases = basename[:int(len(basename) * 0.8)]  # 80%作为训练集
    valid_cases = basename[int(len(basename) * 0.8):]  # 20%作为验证集

    train_data = MaskDataset(train_dir, train_cases, train=True)
    valid_data = MaskDataset(train_dir, valid_cases, train=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_worker, collate_fn = lambda data: (data[0], data[1]))
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=n_worker, collate_fn = lambda data: (data[0], data[1]))

    segment_model = get_maskrcnn_resnet50_fpn_v2(n_class, pretrained_path)
    print('Model:\n', segment_model)
