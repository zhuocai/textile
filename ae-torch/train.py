import torch
from torch import nn, optim

import numpy as np
from baselineCAE import baselineCAE
from tqdm import tqdm

from dataset import get_test_loader, get_trainval_loader
from sklearn.metrics import roc_auc_score

color_mode = 'gray_scale'
lr = 0.1
device = 'cuda'
epoch = 10
tmax = epoch


def calculate_loss(images, outputs, mode):
    e = outputs['e']
    d = outputs['d']
    o = outputs['o']
    d_normed = d / torch.norm(d)
    e_normed = e / torch.norm(e)
    loss_e = torch.mean(torch.square(images - o))
    # loss_e = -torch.mean(images * o/(images.std() * o.std() + 1e-8))
    loss_h = torch.mean(torch.square(d_normed - e_normed))
    if mode == 'e':
        return loss_e
    elif mode == 'h':
        return loss_h
    elif mode == 'eh':
        return loss_e + loss_h


def get_score(images, images_recon, method='l2'):
    n = images.size(0)
    # score = -torch.mean(((images * images_recon) / (images.std() * images_recon.std() + 1e-8)).view(n, -1), dim=1)
    score = torch.mean((images - images_recon).view(n, -1), dim=1)
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
            # print('output requires grad', outputs['o'].requires_grad)
            loss = calculate_loss(images, outputs, mode)
            loss.backward()
            # print('network grad', model.flatten[0].weight.grad)
            # print('grad norm', torch.std(outputs['o'].grad))
            optimizer.step()

            # print('images', 'min=%.4f, max=%.4f, median=%.4f, std=%.4f' %( images.min().item(), 
#                 images.max().item(), torch.median(images).item(), images.std().item() ))
            
            # print('images rec', 'min=%.4f, max=%.4f, median=%.4f, std=%.4f' %( outputs['o'].min().item(), outputs['o'].max().item(), 
 #                   outputs['o'].median().item(), outputs['o'].std().item() ))
            train_loss += loss.item()
            train_bar.set_postfix_str('batch=%d, loss=%.4f' % (batch_idx, train_loss / (batch_idx + 1)))
        scheduler.step()
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
    all_labels = []
    all_images = []
    all_images_recon = []
    data_bar = tqdm(enumerate(loader))
    for idx, (images, labels) in data_bar:
        if color_mode == 'gray_scale':
            images = torch.mean(images, dim=1, keepdim=True)
        images = images.to(device)
        outputs = model(images)
        images_recon = outputs['o']
        all_images.append(to_numpy(images))
        all_images_recon.append(to_numpy(images_recon))
        score = get_score(images, images_recon)
        scores.append(score)
        all_labels.append(labels)

    scores = torch.cat(scores).cpu().numpy()
    all_labels = to_numpy(torch.cat(all_labels))
    all_images = np.concatenate(all_images)
    all_images_recon = np.concatenate(all_images_recon)

    np.save('test_images.npy', all_images)
    np.save('test_images_recon.npy', all_images_recon)
    np.save('test_scores.npy', scores)
    np.save('test_labels.npy', all_labels)
    return scores, all_labels


def get_auroc_val(inlier_score, outlier_score):
    scores = np.concatenate((inlier_score, outlier_score))
    true = np.concatenate((np.zeros_like(inlier_score), np.ones_like(outlier_score)))
    return roc_auc_score(true, scores)

def to_numpy(tens):
    return tens.detach().cpu().numpy()

def check_data(loader):
    all_labels = []
    for (images, labels) in loader:
        all_labels.append(labels)
    all_labels = to_numpy(torch.cat(all_labels))
    for i in range(10):
        print('label = %d, #=%d' % (i, np.sum(all_labels==i)))
    return all_labels

if __name__ == '__main__':
    model = baselineCAE(color_mode=color_mode)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)

    train_loader, val_loader = get_trainval_loader()
    test_loader = get_test_loader()
    check_data(test_loader)
    train(model, train_loader, val_loader, optimizer, lr_scheduler=scheduler,
          epochs=epoch, mode='e')
    # train_score = predict(model, train_loader)
    test_score, test_labels = predict(model, test_loader)
    # np.save('train_score.npy', train_score)
    #np.save('test_score.npy', test_score)
    #np.save('test_labels.npy', test_labels) 
    print(roc_auc_score((test_labels!=2).astype(float), test_score))
