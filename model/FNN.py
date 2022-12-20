import torch
from layer.layers import FeatureEmbedding, DenseLayer
from utils.criteo import criteoDataSet
from FM import FM
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class FNN(torch.nn.Module):
    def __init__(self, feature_nums, emb_dims):
        super(FNN, self).__init__()
        self.emb1 = FeatureEmbedding(feature_nums, 1)
        self.emb2 = FeatureEmbedding(feature_nums, emb_dims)

        self.mlp = DenseLayer([len(feature_nums)*(emb_dims+1), 128, 64, 32, 1])

    def forward(self, x):
        w = self.emb1(x).squeeze(-1)
        v = self.emb2(x).reshape(x.shape[0], -1)
        stacked = torch.hstack([w, v])

        output = self.mlp(stacked)
        return torch.sigmoid(output)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = FNN(field_dims, EMBEDDING_DIM)

fm = FM(field_dims, EMBEDDING_DIM)
fm.load_state_dict(torch.load('./weight/FM.pkl'))
# 参数赋值,利用预先训练好的FM模型初始化第一层embedding的参数向量
fm_state_dict = fm.state_dict()
fnn_state_dict = model.state_dict()
fnn_state_dict['emb1.embedding.weight'] = fm_state_dict['linear.embedding.weight']
fnn_state_dict['emb2.embedding.weight'] = fm_state_dict['emb.embedding.weight']
fnn_state_dict['mlp.mlp.0.bias'] = torch.zeros_like(fnn_state_dict['mlp.mlp.0.bias']).fill_(fm_state_dict['bias'].item())
model.load_state_dict(fnn_state_dict)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# for name, parameter in model.named_parameters():
#     print(name, ':', parameter)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='FNN', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))
