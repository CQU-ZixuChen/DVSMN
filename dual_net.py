import torch
from thop import profile
from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import argparse
from Dataset import MyGraphDataset

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'

if torch.cuda.is_available():
    args.device = 'cuda:0'


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class DualNet(torch.nn.Module):
    def __init__(self, args):
        super(DualNet, self).__init__()
        # XJTU
        self.conv1 = ChebConv(1025, 512, 2)
        self.conv2 = ChebConv(512, 512, 2)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)

        self.lin01 = torch.nn.Linear(1024, 512)
        self.lin02 = torch.nn.Linear(512, 256)
        self.lin03 = torch.nn.Linear(256, 2)
        self.bn01 = torch.nn.BatchNorm1d(512)
        self.bn02 = torch.nn.BatchNorm1d(256)

        self.lin11 = torch.nn.Linear(1024, 512)
        self.lin12 = torch.nn.Linear(512, 256)
        self.lin13 = torch.nn.Linear(256, 5)
        self.bn11 = torch.nn.BatchNorm1d(512)
        self.bn12 = torch.nn.BatchNorm1d(256)

    def forward(self, data, flag, feature_s):
        x, edge_index, batch, A = data.x, data.edge_index, data.batch, data.A
        x = F.relu(self.bn1(self.conv1(x, edge_index)))

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        if flag == 0:
            prob = torch.randn(len(x), 1).to(args.device)
            miu = torch.mean(x, dim=1).unsqueeze(1).to(args.device)
            sigma = torch.std(x, dim=1).unsqueeze(1).to(args.device)
            s = sigma * (1 + prob)
            m = miu * (1 + prob)
            x = s * (x - miu) / sigma + m
        if flag == 1:
            prob = torch.randn(len(x), 1).to(args.device)
            miu_s = torch.mean(feature_s, dim=1).unsqueeze(1).to(args.device)
            sigma_s = torch.std(feature_s, dim=1).unsqueeze(1).to(args.device)
            miu_t = torch.mean(x, dim=1).unsqueeze(1).to(args.device)
            sigma_t = torch.std(x, dim=1).unsqueeze(1).to(args.device)
            s = prob * sigma_s + (1-prob) * sigma_t
            m = prob * miu_s + (1-prob) * miu_t
            x = s * (x - miu_t) / sigma_t + m

        feature = x
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        feature0 = x
        # Binary output
        x0 = F.relu(self.bn01(self.lin01(x)))
        x0 = F.dropout(x0, p=0.5, training=self.training)

        x0 = F.relu(self.bn02(self.lin02(x0)))
        x0 = F.dropout(x0, p=0.5, training=self.training)

        x0 = F.relu(self.lin03(x0))
        output0 = F.softmax(x0, dim=-1)
        # Class-wise output
        x1 = F.relu(self.bn11(self.lin11(x)))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x1 = F.relu(self.bn12(self.lin12(x1)))
        x1 = F.dropout(x1, p=0.5, training=self.training)
        feature1 = x1

        x1 = F.relu(self.lin13(x1))
        output1 = F.softmax(x1, dim=-1)

        return output0, output1, feature, feature0, feature1


def main():
    dataset = MyGraphDataset('Name of your graph dataset')
    data = dataset[0]
    data = data.to(args.device)
    model = DualNet(args).to(args.device)
    # When source domain data is propagated forward, flag = 0
    # When target domain data is propagated forward, flag = 1
    # feature_s is the self-mixed source domain feature
    output0, output1, feature, feature0, feature1 = model(data, flag, feature_s)


if __name__ == '__main__':
     main()




