import torch
from utils.Trainer import Trainer
from utils.amazon import AmazonBooksDataset
from layer.layers import Embedding, DenseLayer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class BaseModel(torch.nn.Module):
    def __init__(self, feature_nums, emb_dim):
        super(BaseModel, self).__init__()

        self.emb = Embedding(feature_nums[0], emb_dim)
        self.mlp = DenseLayer([emb_dim*2, 120, 60, 1])

    def forward(self, x):
        user_behaviors = x[:, :-1]

        # avg+sum pooling操作
        mask = (user_behaviors > 0).float().unsqueeze(-1)
        avg = mask.mean(dim=1, keepdim=True)
        weight = mask.mul(avg)
        user_behavior_embedding = self.emb(user_behaviors).mul(weight).sum(dim=1)

        ad_embedding = self.emb(x[:, -1])

        concated = torch.hstack([user_behavior_embedding, ad_embedding])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = AmazonBooksDataset("../data/amazon-books-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = BaseModel(field_dims, EMBEDDING_DIM)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# for name, parameter in model.named_parameters():
#     print(name, ':', parameter)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='DIN_Base', epoch=EPOCH, trials=100, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))