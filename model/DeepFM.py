import torch
from layer.layers import FeatureEmbedding, FeatureInteraction, DenseLayer
from utils.criteo import criteoDataSet
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class DeepFM(torch.nn.Module):
    def __init__(self, feature_nums, emb_dims):
        super(DeepFM, self).__init__()
        self.linear = FeatureEmbedding(feature_nums, 1)
        self.emb = FeatureEmbedding(feature_nums, emb_dims)
        self.DeepCross = FeatureInteraction()
        self.deep = DenseLayer([emb_dims * len(feature_nums), 128, 64, 32])
        self.fc = torch.nn.Linear(1 + 32 + len(feature_nums)*(len(feature_nums)-1)//2, 1)

    def forward(self, x):
        lr = self.linear(x)
        embedding = self.emb(x)
        fm = self.DeepCross(embedding).sum(dim=2)
        
        deep = self.deep(embedding.reshape(x.shape[0], -1))
        
        stacked = torch.hstack([lr.sum(dim=1), fm, deep])
        output = self.fc(stacked)
        return torch.sigmoid(output)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = DeepFM(field_dims, EMBEDDING_DIM)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='DeepFM', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))