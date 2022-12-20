import torch
from layer.layers import FeatureEmbedding
from utils.movieLens import movieLens
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class LFM(torch.nn.Module):
    def __init__(self, feature_nums, emd_dim):
        super(LFM, self).__init__()
        self.embed = FeatureEmbedding(feature_nums, emd_dim)

    def forward(self, x):
        embedding = self.embed(x)
        output = torch.mul(embedding[:, 0], embedding[:, 1]).sum(dim=1, keepdim=True)
        return torch.sigmoid(output)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = movieLens("../data/ml-latest-small-ratings.txt").to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = LFM(field_dims, EMBEDDING_DIM)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
    train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
    train.train(train_X, train_Y, name='LFM', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
    loss, metrics = train.test(test_X, test_Y)
    print("test loss:{:.5f} | test metric:{:.5f}".format(loss, metrics))


