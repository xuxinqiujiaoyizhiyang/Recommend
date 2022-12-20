# https://zhuanlan.zhihu.com/p/91057914
import torch
from utils.Trainer import Trainer
from utils.criteo import criteoDataSet
from layer.layers import FeatureEmbedding

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class ResidualUnit(torch.nn.Module):
    def __init__(self, input_size):
        super(ResidualUnit, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, input_size)
        self.fc2 = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(torch.relu(output))
        return x + output

class DeepCrossing(torch.nn.Module):
    def __init__(self, feature_nums, emb_dims, res_num=1):
        super(DeepCrossing, self).__init__()
        input_size = len(feature_nums) * emb_dims
        self.emb = FeatureEmbedding(feature_nums, emb_dims)

        self.ResList = torch.nn.Sequential(*[ResidualUnit(input_size) for _ in range(res_num)])
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        embedding = self.emb(x).reshape(x.shape[0], -1)
        res = self.ResList(embedding)
        output = self.fc(res)
        return torch.sigmoid(output)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = DeepCrossing(field_dims, EMBEDDING_DIM)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='DeepCrossing', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))
