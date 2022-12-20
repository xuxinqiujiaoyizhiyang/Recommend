import torch
from utils.Trainer import Trainer
from utils.criteo import criteoDataSet
from layer.layers import FeatureEmbedding, FeatureInteraction

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class AttentionNet(torch.nn.Module):
    def __init__(self, emb_dims, t=4):
        super(AttentionNet, self).__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(emb_dims, t),
            torch.nn.ReLU(),
            torch.nn.Linear(t, 1, bias=False),
            torch.nn.Flatten(), # (b, feature_cross)
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.attention(x)

class AFM(torch.nn.Module):
    def __init__(self, feature_nums, emb_dims):
        super(AFM, self).__init__()
        self.linear = FeatureEmbedding(feature_nums, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1, )))

        self.emb = FeatureEmbedding(feature_nums, emb_dims)
        self.Interaction = FeatureInteraction()
        self.Attention = AttentionNet(emb_dims)
        self.p = torch.nn.Parameter(torch.zeros(emb_dims, ))
        torch.nn.init.xavier_uniform_(self.p.unsqueeze(-1).data)

    def forward(self, x):
        embedding = self.emb(x)
        # attention part
        interaction = self.Interaction(embedding)
        att = self.Attention(interaction)
        att_part = torch.mul(torch.mul(att.unsqueeze(-1), interaction).sum(dim=1), self.p).sum(dim=1).unsqueeze(-1)
        
        output = self.bias + self.linear(x).sum(dim=1) + att_part
        return torch.sigmoid(output)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = AFM(field_dims, EMBEDDING_DIM)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# for name, parameter in model.named_parameters():
#     print(name, ':', parameter)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='AFM', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))