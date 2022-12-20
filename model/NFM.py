import torch
from layer.layers import FeatureEmbedding, FeatureInteraction, DenseLayer
from model.utils.Trainer import Trainer
from model.utils.criteo import criteoDataSet

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class NFM(torch.nn.Module):
    def __init__(self, feature_nums, emb_dims):
        super(NFM, self).__init__()
        self.linear = FeatureEmbedding(feature_nums, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1, )))

        self.emb = FeatureEmbedding(feature_nums, emb_dims)
        self.Interaction = FeatureInteraction()
        self.mlp = DenseLayer([emb_dims, 256, 128, 1])

    def forward(self, x):
        lr = self.linear(x).sum(dim=1)
        # Bi-Interaction pooling,fm的二维特征交叉的变形形式
        embedding = self.emb(x)
        interaction = self.Interaction(embedding).sum(dim=1)
        deep = self.mlp(interaction)

        output = lr + self.bias + deep
        return torch.sigmoid(output)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = NFM(field_dims, EMBEDDING_DIM)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='NFM', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))

