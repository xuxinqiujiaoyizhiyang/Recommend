import torch
import numpy as np


# 在 cpu 下，比 nn.Embedding 快，但是在 gpu 的序列模型下比后者慢太多了
class cpuEmbedding(torch.nn.Module):
    def __init__(self, feature_nums, emd_layers):
        super(cpuEmbedding, self).__init__()
        self.w = torch.nn.Parameter(torch.zeros(feature_nums, emd_layers))
        torch.nn.init.xavier_normal_(self.w.data)

    def forward(self, x):
        return self.w[x]

# Embedding的集成类，取决于cuda是否可用
class Embedding:
    def __new__(cls, feature_nums, emd_layers):
        if torch.cuda.is_available():
            embedding = torch.nn.Embedding(feature_nums, emd_layers)
            torch.nn.init.xavier_normal_(embedding.weight.data)
            return embedding
        else:
            return cpuEmbedding(feature_nums, emd_layers)

# featureEmbedding,w(特征域下的位置+offsets)
class FeatureEmbedding(torch.nn.Module):
    def __init__(self, field_dims, emd_layers):
        super(FeatureEmbedding, self).__init__()
        self.embedding = Embedding(sum(field_dims), emd_layers)
        self.offsets = torch.tensor([0, *np.cumsum(field_dims)[:-1]], dtype=torch.long)

    def forward(self, x):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        x = x.clone().detach().long().to(device)
        x = x + self.offsets.to(device)
        return self.embedding(x)

# 特征交叉
class FeatureInteraction(torch.nn.Module):
    def __init__(self):
        super(FeatureInteraction, self).__init__()

    def forward(self, x):
        feature_nums = x.shape[1]
        one = []
        two = []
        for i in range(feature_nums):
            for j in range(i + 1, feature_nums):
                one.append(i)
                two.append(j)

        res = torch.mul(x[:, one], x[:, two])
        return res

# 深层网络层
class DenseLayer(torch.nn.Module):
    def __init__(self, layer, batch_norm=True):
        super(DenseLayer, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layers[1: -1]:
            layers.append(torch.nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(output_size))
            layers.append(torch.nn.ReLU(inplace=True))
            input_size = output_size

        layers.append(torch.nn.Linear(input_size, layer[-1]))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)