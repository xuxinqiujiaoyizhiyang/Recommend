import torch
from layer.layers import FeatureEmbedding, DenseLayer
from utils.criteo import criteoDataSet
from utils.Trainer import Trainer

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300

class Cross(torch.nn.Module):
    def __init__(self, width):
        super(Cross, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros((1, )))
        self.w = torch.nn.Parameter(torch.zeros(width, ))
        # 初始化的时候转化为二维才能顺利进行
        torch.nn.init.xavier_uniform_(self.w.unsqueeze(0).data)

    def forward(self, x0x1):
        x0, x1 = x0x1
        x2 = torch.mul(x0, torch.mul(x1, self.w).sum(dim=1, keepdim=True)) + self.bias + x1
        return x0, x2

class CrossNet(torch.nn.Module):
    def __init__(self, width, layer_num):
        super(CrossNet, self).__init__()
        self.ModuleList = torch.nn.Sequential(*[Cross(width) for _ in range(layer_num)])

    def forward(self, x0):
        _, output = self.ModuleList((x0, x0))
        return output

class DCN(torch.nn.Module):
    def __init__(self, feature_nums, emb_nums, layer_nums=3):
        super(DCN, self).__init__()
        self.emb = FeatureEmbedding(feature_nums, emb_nums)
        self.CrossNet = CrossNet(len(feature_nums)*emb_nums, layer_nums)
        self.deep = DenseLayer([len(feature_nums)*emb_nums, 128, 64, 32])
        self.fc = torch.nn.Linear(32 + len(feature_nums)*emb_nums, 1, bias=True)

    def forward(self, x):
        embedding = self.emb(x).reshape(x.shape[0], -1)
        cross = self.CrossNet(embedding)
        deep = self.deep(embedding)
        stack = torch.hstack([cross, deep])
        output = self.fc(stack)
        return torch.sigmoid(output)
    

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = criteoDataSet("../data/criteo-100k.txt")
data.to(device)

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.train_valid_test_split()
model = DCN(field_dims, EMBEDDING_DIM)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# for name, parameter in model.named_parameters():
#     print(name, ':', parameter)

if __name__ == '__main__':
     Train = Trainer(model, criterion, optimizer, batch_size=BATCH_SIZE, device=device)
     Train.train(train_X, train_Y, name='DCN', epoch=EPOCH, trials=30, valid_x=valid_X, valid_y=valid_Y)
     test_loss, test_metric = Train.test(test_X, test_Y)
     print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))