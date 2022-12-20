import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from utils.criteo import criteoDataSet
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils.Trainer import Trainer
from layer.layers import FeatureEmbedding

EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 300
TRIAL = 100

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def transform(lgbm, train_X, train_y, valid_X, valid_y, test_X, test_y):
    onehot = OneHotEncoder()
    onehot_data = onehot.fit_transform(np.vstack([train_X.cpu(), valid_X.cpu(), test_X.cpu()])).A
    train_len, valid_len, test_len = len(train_X), len(valid_X), len(test_X)
    sparse_train_X = onehot_data[:train_len]
    sparse_valid_X = onehot_data[train_len: -test_len]
    sparse_test_X = onehot_data[-test_len:]
    # lgbm fit transoform
    lgbm.fit(sparse_train_X, train_y.cpu().ravel(),
             eval_set=[(sparse_valid_X, valid_y.cpu().ravel())],
             callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=100)])
    fusion_train_X = np.hstack([lgbm.predict(sparse_train_X, pred_leaf=True), train_X.cpu()])
    fusion_valid_X = np.hstack([lgbm.predict(sparse_valid_X, pred_leaf=True), valid_X.cpu()])
    fusion_test_X = np.hstack([lgbm.predict(sparse_test_X, pred_leaf=True), test_X.cpu()])
    # 原始特征+分类特征每一维度的最大位置取值
    fusion_field_dims = (np.vstack([fusion_train_X, fusion_valid_X, fusion_test_X]).astype(np.int32).max(axis=0) + 1).tolist()

    fusion_train_X = torch.tensor(fusion_train_X, dtype=torch.long).to(device)
    fusion_valid_X = torch.tensor(fusion_valid_X, dtype=torch.long).to(device)
    fusion_test_X = torch.tensor(fusion_test_X, dtype=torch.long).to(device)
    return fusion_field_dims, fusion_train_X, fusion_valid_X, fusion_test_X

class LogisticRegression(torch.nn.Module):
    def __init__(self, feature_nums):
        super(LogisticRegression, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros((1, )))
        self.linear = FeatureEmbedding(feature_nums, 1)

    def forward(self, x):
        output = self.linear(x).sum(dim=1) + self.bias
        return torch.sigmoid(output)

lgbm = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=10320,
    num_leaves=32,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    metric='auc',
    objective='binary'
)

data = criteoDataSet("../data/criteo-100k.txt")

field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = data.to(device).train_valid_test_split()
fusion_dims, train_X, valid_X, test_X = transform(lgbm, train_X, train_Y, valid_X, valid_Y, test_X, test_Y)

model = LogisticRegression(fusion_dims)
model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

if __name__ == '__main__':
    Train = Trainer(model, criterion, optimizer, BATCH_SIZE, device=device)
    Train.train(train_X, train_Y, 'GBDT+LR', EPOCH, trials=True, valid_x=valid_X, valid_y=valid_Y)

    test_loss, test_metric = Train.test(test_X, test_Y)
    print("test_loss:{0:.5f} | test_metric:{1:.5f}".format(test_loss, test_metric))