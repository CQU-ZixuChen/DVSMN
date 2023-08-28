import torch
import torch.nn.functional as F
from sklearn import metrics
import math
from sklearn.metrics import pairwise_distances
import numpy as np
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'


def cal_loss(model, data1, data2):
    output0, output1, feature_s, feature0, feature1 = model(data1, 0, 0)
    output2 = torch.cat((output1[:, 0].unsqueeze(1), torch.sum(output1[:, 1:4], dim=1).unsqueeze(1)), 1)
    weight = 2 * torch.sigmoid(1 - F.cosine_similarity(output0, output2, dim=1))

    y0 = F.one_hot(data1.y1, num_classes=2)
    loss0 = - y0 * torch.log(output0 + 1e-6)
    loss0 = torch.sum(loss0) / y0.shape[0]

    y1 = F.one_hot(data1.y, num_classes=5)
    loss1 = - y1 * torch.log(output1 + 1e-6)
    loss1 = torch.sum(weight * torch.sum(loss1, dim=1)) / y1.shape[0]

    output_t0, output_t1, feature_t, feature_t0, feature_t1 = model(data2, 1, feature_s)

    miu0 = torch.sum(torch.mean(feature0, dim=1).unsqueeze(1))
    miu1 = torch.sum(torch.mean(feature_t0, dim=1).unsqueeze(1))
    sigma0 = torch.sum(torch.std(feature0, dim=1).unsqueeze(1))
    sigma1 = torch.sum(torch.std(feature_t0, dim=1).unsqueeze(1))
    lossDA = torch.abs(miu0-miu1) + torch.abs(sigma0-sigma1)

    loss = 0.5 * loss0 + 0.5 * loss1 + lossDA
    return loss, loss0, loss1, lossDA


def tst(model, loader):
    model.eval()
    correct = 0.
    for data in loader:
        data = data.to(args.device)
        output0, output1, feature, feature0, feature1 = model(data, 2, 0)
        output2 = torch.cat((output1[:, 0].unsqueeze(1), torch.sum(output1[:, 1:4], dim=1).unsqueeze(1)), 1)
        weight = 2 * torch.sigmoid(1 - F.cosine_similarity(output0, output2, dim=1))

        y0 = F.one_hot(data.y1, num_classes=2)
        loss0 = - y0 * torch.log(output0 + 1e-6)
        loss0 = torch.sum(loss0) / y0.shape[0]

        y1 = F.one_hot(data.y, num_classes=5)
        loss1 = - y1 * torch.log(output1 + 1e-6)
        loss1 = torch.sum(weight * torch.sum(loss1, dim=1)) / y1.shape[0]

        feature1 = feature1.detach()
        feature1 = feature1.cpu().numpy()

        output1 = output1.max(dim=1)[1]
        correct += output1.eq(data.y).sum().item()
        accuracy = correct / len(data.y)

        output1 = output1.detach()
        output1 = output1.cpu().numpy()

        data.y = data.y.cpu().numpy()
    return _, accuracy, feature, feature1, output1, data.y
