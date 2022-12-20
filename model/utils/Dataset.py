import torch
from sklearn.model_selection import train_test_split
# dataset的通用类
class Dataset:
    def __init__(self):
        self.device = torch.device('cpu')

    def to(self, device):
        self.device = device
        return self

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        # 每一维特征向量的类别取值个数
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        # 分割训练集、验证集和测试集
        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2022)
        valid_size = valid_size / (valid_size + test_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2022)
        device = self.device

        # 分割所需训练数据，验证数据和测试数据
        train_X, train_Y = torch.tensor(train[:, :-1], dtype=torch.float).to(device), torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_X, valid_Y = torch.tensor(valid[:, :-1], dtype=torch.float).to(device), torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_X, test_Y = torch.tensor(test[:, :-1], dtype=torch.float).to(device), torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)