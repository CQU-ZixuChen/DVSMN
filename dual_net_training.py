import torch
import scipy.io as scio
import numpy as np
from function import mmd_rbf, tst, cal_loss
from torch_geometric.loader import DataLoader
from dual_net import DualNet
import argparse
import time
from Dataset import MyGraphDataset
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')

args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'

# Imbalance Ratio Index
char1 = ['10%', '20%', '40%', '60%', '80%', '100%']
char2 = ['10%', '20%', '40%', '60%', '80%', '100%']

for c1, c2 in zip(char1, char2):
    dataset1 = MyGraphDataset("name of the source graph dataset")
    dataset2 = MyGraphDataset("name of the target graph dataset")
    dataset3 = MyGraphDataset("name of the test graph dataset")
    # Each task is repeated 10 trials
    for n in range(10):
        train_loader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
        train_loader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset3, batch_size=int(len(dataset3)), shuffle=False)

        model = DualNet(args).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        min_loss = 1e10
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(zip(train_loader1, train_loader2)):
                start_train = time.time()
                data1 = data[0].to(args.device)
                data2 = data[1].to(args.device)
                loss, loss0, loss1, lossDA = cal_loss(model, data1, data2)
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if loss < min_loss:
                torch.save(model.state_dict(), 'latest.pth')
                min_loss = loss
                patience = 0
            else:
                patience += 1
        model = DualNet(args).to(args.device)
        model.load_state_dict(torch.load('latest.pth'))
        test_loss1, test_acc1, feature, feature1, output1, label1 = tst(model, test_loader)
        print("Test accuarcy:{}".format(test_acc1))












