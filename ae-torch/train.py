import torch
from torch import nn, optim

import numpy as np
from baselineCAE import baselineCAE
from tqdm import tqdm

from dataset import get_test_loader, get_trainval_loader
from sklearn.metrics import roc_auc_score

color_mode = 'gray_scale'
lr = 0.1
device = 'cpu'
epoch = 10
tmax = epoch


def calculate_loss(images, outputs, mode):
    e = outputs['e']
    d = outputs['d']
    o = outputs['o']
    d_normed = d / torch.norm(d)
    e_normed = e / torch.norm(e)
    loss_e = torch.sum(torch.square(images - o))
    loss_h = torch.sum(torch.square(d_normed - e_normed))
    if mode == 'e':
        return loss_e
    elif mode == 'h':
        return loss_h
    elif mode == 'eh':
        return loss_e + loss_h


def get_score(images, images_recon, method='l2'):
    n = images.size(0)
    score = torch.sum(torch.square((images - images_recon)).view(n, -1), dim=1)
    return score.detach()


def train(model: nn.Module, train_loader, val_loader, optim, lr_scheduler, epochs, mode='eh'):
    model.train()
    # mode = 'eh' or 'h' or 'e'

    for epoch in range(epochs):
        train_loss = 0
        train_bar = tqdm(enumerate(train_loader))
        model.train()
        for batch_idx, (images, labels) in train_bar:
            if color_mode == 'gray_scale':
                images = torch.mean(images, dim=1, keepdim=True)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = calculate_loss(images, outputs, mode)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix_str('batch=%d, loss=%.4f' % (batch_idx, train_loss / (batch_idx + 1)))

        model.eval()
        val_loss = 0
        val_loss_aver = 0
        val_bar = tqdm(enumerate(val_loader))
        for batch_idx, (images, labels) in val_bar:
            if color_mode == 'gray_scale':
                images = torch.mean(images, dim=1, keepdim=True)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = calculate_loss(images, outputs, mode)
            val_loss += loss.item()
            val_loss_aver = val_loss / (batch_idx + 1)
        print('val_loss=%.4f' % (val_loss_aver))


def predict(model, loader, mode='e', color_mode='gray_scale'):
    model.eval()

    scores = []

    data_bar = tqdm(enumerate(loader))
    for idx, (images, labels) in data_bar:
        if color_mode == 'gray_scale':
            images = torch.mean(images, dim=1, keepdim=True)
        images = images.to(device)
        outputs = model(images)
        images_recon = outputs['o']
        score = get_score(images, images_recon)
        scores.append(score)

    scores = torch.cat(scores).cpu().numpy()
    return scores


def get_auroc_val(inlier_score, outlier_score):
    scores = np.concatenate((inlier_score, outlier_score))
    true = np.concatenate((np.zeros_like(inlier_score), np.ones_like(outlier_score)))
    return roc_auc_score(true, scores)


if __name__ == '__main__':
    model = baselineCAE(color_mode=color_mode)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)

    train_loader, val_loader = get_trainval_loader()
    test_loader = get_test_loader()
    train(model, train_loader, val_loader, optimizer, lr_scheduler=scheduler,
          epochs=epoch, mode='e')
    train_score = predict(model, train_loader)
    test_score = predict(model, test_loader)
    print(roc_auc_score(train_score, test_score))
