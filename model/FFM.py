import torch
from layer.layers import FeatureEmbedding
from utils.criteo import criteoDataSet
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class FFM(torch.nn.Module):
    def __init__(self, feature_nums, emd):
        super(FFM, self).__init__()
        self.feature_nums =feature_nums
        self.ModuleList = torch.nn.ModuleList([FeatureEmbedding(feature_nums, emd) for _ in feature_nums])
        self.linear = FeatureEmbedding(feature_nums, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        feature_len = len(self.feature_nums)

        embedding = [emb(x) for emb in self.ModuleList]
        # 在feature维度进行了拼接，这里注意一下hsatck的拼接维度
        embedding = torch.hstack(embedding)

        i_list = []
        j_list =[]
        for i in range(feature_len):
            for j in range(i+1, feature_len):
                i_list.append(j * feature_len + i)
                j_list.append(i * feature_len + j)
        # 先进行emd维度的加和，类似于向量的矩阵相乘（这里通过mul和add的操作进行实现），之后进行的是不同特征组合的上一步的目标值的加和
        interaction = torch.mul(embedding[:, i_list], embedding[:, j_list]).sum(dim=2).sum(dim=1).unsqueeze(-1)
        # 通过embdding的计算形式实现linear层，有效的解决了维度爆炸的问题
        FFM = self.linear(x).sum(dim=1) + self.bias + interaction
        return torch.sigmoid(FFM)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = FFM(field_dims, EMBEDDING_DIM)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
    train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
    train.train(train_X, train_Y, name='FFM', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
    loss, metrics = train.test(test_X, test_Y)
    print("test loss:{:.5f} | test metric:{:.5f}".format(loss, metrics))


