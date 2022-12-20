import torch
from utils.movieLens import movieLens
from utils.Trainer import Trainer
from layer.layers import FeatureEmbedding, DenseLayer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 100
TRIAL = 100

class NeuralCF(torch.nn.Module):
    def __init__(self, feature_nums, emb_dim):
        super(NeuralCF, self).__init__()
        self.embGMF = FeatureEmbedding(feature_nums, emb_dim)
        self.embMLP = FeatureEmbedding(feature_nums, emb_dim)

        self.mlp = DenseLayer([len(feature_nums)*emb_dim, 128, 64])
        self.fc = torch.nn.Linear(64 + emb_dim, 1, bias=True)

    def forward(self, x):
        embedding_GMF = self.embGMF(x)
        GMF = torch.mul(embedding_GMF[:, 0], embedding_GMF[:, 1]).squeeze(-1)

        embedding_MLP = self.embMLP(x)
        MLP = self.mlp(embedding_MLP.reshape(embedding_MLP.shape[0], -1))
        input = torch.hstack([GMF, MLP])
        output = self.fc(input)
        return torch.sigmoid(output)

data = movieLens('../data/ml-latest-small-ratings.txt')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = NeuralCF(field_dims, EMBEDDING_DIM)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='NeuralCF', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))