import torch
from utils.criteo import criteoDataSet
from layer.layers import FeatureEmbedding, DenseLayer
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class WideDeep(torch.nn.Module):
    def __init__(self, feature_nums, emb_nums):
        super(WideDeep, self).__init__()
        self.wide = FeatureEmbedding(feature_nums, 1)

        self.emb = FeatureEmbedding(feature_nums, emb_nums)
        self.deep = DenseLayer([len(feature_nums)*emb_nums, 256, 128, 64], batch_norm=True)
        # 这里使用的是wide+deep（最经典），其实wide部分可以更换为正常的embedding向量
        self.linear = torch.nn.Linear(64 + len(feature_nums), 1, bias=True)


    def forward(self, x):
        wide = self.wide(x).view(x.shape[0], -1)

        embedding = self.emb(x).view(x.shape[0], -1)
        deep = self.deep(embedding)

        wideDeep = torch.hstack([wide, deep])
        output = self.linear(wideDeep)
        return torch.sigmoid(output)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = WideDeep(field_dims, EMBEDDING_DIM)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='WideDeep', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))