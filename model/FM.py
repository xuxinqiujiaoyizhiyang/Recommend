import torch
from utils.criteo import criteoDataSet
from layer.layers import FeatureEmbedding
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class FM(torch.nn.Module):
    def __init__(self, feature_nums, emb_nums):
        super(FM, self).__init__()
        self.emb = FeatureEmbedding(feature_nums, emb_nums)
        self.linear = FeatureEmbedding(feature_nums, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1, )))


    def forward(self, x):
        embedding = self.emb(x)
        sum_square = embedding.sum(dim=1).pow(2)
        square_sum = embedding.pow(2).sum(dim=1)
        interation = (sum_square - square_sum).sum(dim=1).unsqueeze(-1)
        fm = self.linear(x).sum(dim=1) + self.bias + interation / 2
        return torch.sigmoid(fm)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = FM(field_dims, EMBEDDING_DIM)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='FM', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))