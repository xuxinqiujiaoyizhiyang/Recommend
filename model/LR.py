import torch
from layer.layers import FeatureEmbedding
from utils.criteo import criteoDataSet
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class LR(torch.nn.Module):
    def __init__(self, feature_nums):
        super(LR, self).__init__()
        self.linear = FeatureEmbedding(feature_nums, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        LR = self.linear(x).sum(dim=1) + self.bias
        return torch.sigmoid(LR)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = LR(field_dims)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
    train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
    train.train(train_X, train_Y, name='LR', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
    loss, metrics = train.test(test_X, test_Y)
    print("test loss:{:.5f} | test metric:{:.5f}".format(loss, metrics))


