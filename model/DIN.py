import torch
from utils.Trainer import Trainer
from utils.amazon import AmazonBooksDataset
from layer.layers import Embedding

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 100

class Dice(torch.nn.Module):

    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x)

        return x.mul(p) + self.alpha * x.mul(1 - p)


class ActivationUnit(torch.nn.Module):

    def __init__(self, embed_dim=4):
        super(ActivationUnit, self).__init__()
        self.mlp = torch.nn.Sequential(
            # torch.nn.Linear(embed_dim * (embed_dim + 2), 36), 外积的线性层维度
            torch.nn.Linear(embed_dim * 3, 36), # 内积的线性层维度
            Dice(),
            torch.nn.Linear(36, 1),
        )

    def forward(self, x):
        behaviors = x[:, :-1]
        num_behaviors = behaviors.shape[1]

        # (batch_size, num_behaviors, embed_dims)
        ads = x[:, [-1] * num_behaviors]

        # outer product
        embed_dim = x.shape[-1]
        i1, i2 = [], []

        for i in range(embed_dim):
            for j in range(embed_dim):
                i1.append(i)
                i2.append(j)
        # 一个列向量乘以一个行向量称作向量的外积，外积是一种特殊的克罗内克积，结果是一个矩阵，即叉乘,这里是外积操作。
        # p = behaviors[:, :, i1].mul(ads[:, :, i2]).reshape(behaviors.shape[0], behaviors.shape[1], -1)

        # 一个行向量乘以一个列向量称作向量的内积，又叫作点积，结果是一个数，即点乘；
        p = behaviors.mul(ads)

        att = self.mlp(torch.cat([behaviors, p, ads], dim=2))
        return att


class DeepInterestNetwork(torch.nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(DeepInterestNetwork, self).__init__()
        # 商品 embedding 层
        self.embed = Embedding(field_dims[0], embed_dim)
        self.attention = ActivationUnit(embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 200),
            Dice(),
            torch.nn.Linear(200, 80),
            Dice(),
            torch.nn.Linear(80, 1)
        )

    def forward(self, x):
        mask = (x > 0).float().unsqueeze(-1)  # (batch_size, num_behaviors+1, 1)
        behaviors_ad_embeddings = self.embed(x).mul(mask)  # (batch_size, num_behaviors+1, embed_dim)
        att = self.attention(behaviors_ad_embeddings)  # (batch_size, num_behaviors, 1)

        weighted_behaviors = behaviors_ad_embeddings[:, :-1].mul(att)  # (batch_size, num_behaviors, embed_dim)
        user_interest = weighted_behaviors.sum(dim=1)  # (batch_size, embed_dim)

        concated = torch.hstack([user_interest, behaviors_ad_embeddings[:, -1]])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = AmazonBooksDataset("../data/amazon-books-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = DeepInterestNetwork(field_dims, EMBEDDING_DIM)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# for name, parameter in model.named_parameters():
#     print(name, ':', parameter)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='DIN', epoch=EPOCH, trials=100, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))
