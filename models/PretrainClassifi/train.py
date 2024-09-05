import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import random
import shutil

from loader import ClassDataset


####
def get_resnet50(n, weights_path=None):
    resnet = models.resnet50()
    if weights_path is not None:
        resnet.load_state_dict(torch.load(weights_path, weights_only=True))
    for parameter in resnet.parameters():
        parameter.requires_grad = False
    cin = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(cin, 1024, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(1024, n, bias=False)
    )

    return resnet


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
    # train_dir = '/tcdata/train/'
    train_dir = '../../tcdata/train/'  # 测试路径
    log_dir = 'log'
    pretrained_path = 'resnet50-11ad3fa6.pth'

    # n_class = 1
    n_class = 3  # 测试时使用3

    # n_epoch = (50, 50)
    n_epoch = (10, 10)  # 测试时使用10+10
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
    train_sample = basename[:int(len(basename) * 0.8)]  # 80%作为训练集
    valid_sample = basename[int(len(basename) * 0.8):]  # 20%作为验证集

    train_data = ClassDataset(
        train_dir, train_sample,
        img_shape=(512, 512),
        mode='train', aug=True)
    valid_data = ClassDataset(
        train_dir, valid_sample,
        img_shape=(512, 512),
        mode='valid', aug=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=n_worker)

    classifi_model = get_resnet50(n_class, pretrained_path)
    print('Model:\n', classifi_model)
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(classifi_model.parameters(), lr=1e-3)

    classifi_model.to(using_device)

    val_loss_all_0 = {}
    for t in range(n_epoch[0]):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, classifi_model, loss_func, optim, using_device)
        val_loss, _ = valid(valid_loader, classifi_model, loss_func, using_device)
        val_loss_all_0['%s/00/epoch_%.3d.pth' % (log_dir, t)] = val_loss
        torch.save(classifi_model.state_dict(), '%s/00/epoch_%.3d.pth' % (log_dir, t))

    best_weight = min(val_loss_all_0, key=lambda k: val_loss_all_0[k])
    print('Load best model:', best_weight)
    classifi_model.load_state_dict(torch.load(best_weight, weights_only=True))

    for param in classifi_model.parameters():
        param.requires_grad = True
    val_loss_all_1 = {}
    for t in range(n_epoch[1]):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, classifi_model, loss_func, optim, using_device)
        val_loss, _ = valid(valid_loader, classifi_model, loss_func, using_device)
        val_loss_all_0['%s/00/epoch_%.3d.pth' % (log_dir, t)] = val_loss
        torch.save(classifi_model.state_dict(), '%s/01/epoch_%.3d.pth' % (log_dir, t))

    best_weight = min(val_loss_all_1, key=lambda k: val_loss_all_1[k])
    print('Best model:', best_weight)
    shutil.copy(best_weight, '%s/best.pth' % log_dir)

    print('###################################################################')
    print('###################################################################')
    print("\nDone!\n")
    print('###################################################################')
    print('###################################################################')
