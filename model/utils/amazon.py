import pandas as pd
from sklearn.model_selection import train_test_split

from model.utils.Dataset import Dataset
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import torch

class AmazonBooksDataset(Dataset):
    def __init__(self, file, read_part=True, sample_nums=100000, sequence_length=40):
        super(AmazonBooksDataset, self).__init__()
        if read_part:
            data_df = pd.read_csv(file, sep=',', nrows=sample_nums)
        else:
            data_df = pd.read_csv(file, sep=',')

        data_df['hist_item_list'] = data_df.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
        data_df['hist_cate_list'] = data_df.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

        # cate_encoder
        cate_list = list(data_df['cateID'])
        data_df.apply(lambda x: cate_list.extend(x['hist_cate_list']), axis=1)
        cate_set = set(cate_list)
        cate_set.add('0')
        cate_encoder = LabelEncoder().fit(list(cate_set))

        hist_limit = sequence_length
        col = ['hist_cate_{}'.format(i) for i in range(hist_limit)]

        def deal(x):
            if len(x) > hist_limit:
                return pd.Series(x[-hist_limit:], index=col)
            else:
                pad = hist_limit - len(x)
                x = x + ['0' for _ in range(pad)]
                return pd.Series(x, index=col)
        # 针对df的某一列进行apply,返回series则可以对列进行扩充
        cate_df = data_df['hist_cate_list'].apply(deal).join(data_df['cateID']).apply(cate_encoder.transform, axis=0).join(data_df['label'])
        self.data = cate_df.values

    def train_valid_test_split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        # 注意和np.max进行区分，np.max中axis默认为0
        field_dims = [self.data[:-1].max().astype(int) + 1]

        # 分割训练集、验证集和测试集
        train, valid_test = train_test_split(self.data, train_size=train_size, random_state=2022)
        valid_size = valid_size / (valid_size + test_size)
        valid, test = train_test_split(valid_test, train_size=valid_size, random_state=2022)
        device = self.device

        # 分割所需训练数据，验证数据和测试数据
        train_X, train_Y = torch.tensor(train[:, :-1], dtype=torch.long).to(device), torch.tensor(train[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        valid_X, valid_Y = torch.tensor(valid[:, :-1], dtype=torch.long).to(device), torch.tensor(valid[:, -1], dtype=torch.float).unsqueeze(1).to(device)
        test_X, test_Y = torch.tensor(test[:, :-1], dtype=torch.long).to(device), torch.tensor(test[:, -1], dtype=torch.float).unsqueeze(1).to(device)

        return field_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)

