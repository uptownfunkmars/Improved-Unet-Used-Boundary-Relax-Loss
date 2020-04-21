import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import UNet
from eval_net import eval_net
from BLR import ImgWtLossSoftNLL
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = r"D:\BaiduNetdiskDownload\gen_img"
dir_mask = r"D:\BaiduNetdiskDownload\gen_mask"
dir_checkpoint = r"D:\PycharmsProjects\UNET+BLR\checkPoints"

classes = 5
epochs = 5
global_step = 0
batch_size = 4
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = BasicDataset(dir_img, dir_mask)
n_val = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

# writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{1}')

net = UNet(n_channels=3, n_classes=classes, bilinear=True)
net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

criterion = ImgWtLossSoftNLL(classes, epochs).cuda()
criterion_ = nn.CrossEntropyLoss().cuda()


for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        imgs = batch['image']
        # print(imgs.size())
        true_masks = batch['mask']
        # print(true_masks.size())
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        masks_pred = net(imgs)
        # masks_pred = masks_pred.to("cpu", torch.double)
        # print(masks_pred.size())

        loss = criterion(masks_pred, true_masks)
        # loss = criterion_(masks_pred, true_masks)
        # print("###########")
        # print(loss.item())
        epoch_loss += loss.item()
        # writer.add_scalar('Loss/train', loss.item(), global_step)

        print("epoch : %d, batch : %5d, loss : %.5f" % (epoch, (global_step / batch_size), loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        if global_step // 1000 == 0 and global_step > 1000:
            val_pre = eval_net(net, val_loader, device)
            print("val loss : %.5f" % val_pre)

    if epoch % 10 == 0 and epoch > 0:
        torch.save(net.state_dict(), dir_checkpoint + f'epoch_%d.pth' % epoch)
