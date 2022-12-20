import torch
from utils.Trainer import Trainer
from utils.criteo import criteoDataSet
from layer.layers import FeatureEmbedding, DenseLayer, FeatureInteraction

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

# 内积
class Inner(torch.nn.Module):
    def __init__(self):
        super(Inner, self).__init__()
        self.Interaction = FeatureInteraction()

    def forward(self, x):
        inner = self.Interaction(x).sum(dim=2)
        return inner
# 外积
class Outer(torch.nn.Module):
    def __init__(self):
        super(Outer, self).__init__()

    def forward(self, x):
        # 外积即向量与向量乘积得到的N*N*M*M的矩阵（N代表特征交叉的数量，M代表emb的维度），为了减少参数，化简计算为N个矩阵的叠加操作：Σf*Σf^T
        p = x.sum(dim=1, keepdims=True)
        # 矩阵维度变换，transpose or permute,前者只能用于二维，mm和matmul前者只能用于二维
        outer = torch.matmul(p.permute(0, 2, 1), p).reshape(x.shape[0], -1)
        return outer

class PNN(torch.nn.Module):
    def __init__(self, feature_nums, emb_dims, method='inner'):
        super(PNN, self).__init__()
        self.emb = FeatureEmbedding(feature_nums, emb_dims)

        if method == 'inner':
            self.pn = Inner()
            input_size = emb_dims*len(feature_nums) + len(feature_nums)*(len(feature_nums) - 1)//2
        elif method == 'outer':
            self.pn = Outer()
            input_size = emb_dims*len(feature_nums) + emb_dims*emb_dims

        self.mlp = DenseLayer([input_size, 256, 1])
        self.bias = torch.nn.Parameter(torch.zeros(1, len(feature_nums)*emb_dims))
        torch.nn.init.xavier_uniform_(self.bias.data)

    def forward(self, x):
        embedding = self.emb(x)
        pn = self.pn(embedding)
        # stacked = torch.cat([embedding.reshape(x.shape[0], -1) + self.bias, pn], dim=1)
        stacked = torch.hstack([embedding.reshape(x.shape[0], -1) + self.bias, pn])
        output = self.mlp(stacked)
        return torch.sigmoid(output)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = PNN(field_dims, EMBEDDING_DIM, method='outer')

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# for name, parameter in model.named_parameters():
#     print(name, ':', parameter)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='PNN-outer', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))