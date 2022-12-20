import torch
from utils.criteo import criteoDataSet
from layer.layers import FeatureEmbedding, DenseLayer
from utils.Trainer import Trainer
# 分片线性的思路，分而治之进行lr模型的训练
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300
class LogisticRegression(torch.nn.Module):
    def __init__(self, feature_nums):
        super(LogisticRegression, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, ))
        self.emb = FeatureEmbedding(feature_nums, 1)

    def forward(self, x):
        output = self.emb(x).sum(dim=1) + self.bias
        return torch.sigmoid(output)

# 将特征分为k类，进行softmax
class Classifier(torch.nn.Module):
    def __init__(self, feature_nums, k):
        super(Classifier, self).__init__()
        self.emb = FeatureEmbedding(feature_nums, k)

    def forward(self, x):
        output = self.emb(x).sum(dim=1)
        return torch.softmax(output, dim=1)

# sigmoid*softmax
class MLR(torch.nn.Module):
    def __init__(self, feature_nums, k=5):
        super(MLR, self).__init__()
        self.clf = Classifier(feature_nums, k)
        self.lr_list = torch.nn.ModuleList([LogisticRegression(feature_nums) for _ in range(k)])

    def forward(self, x):

        clf_list = self.clf(x)
        lr_list = torch.zeros_like(clf_list)

        for i, lr in enumerate(self.lr_list):
            lr_list[:, i] = lr(x).squeeze(-1)

        output = torch.mul(clf_list, lr_list).sum(dim=1, keepdim=True)
        return output

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = MLR(field_dims)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='MLR', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))