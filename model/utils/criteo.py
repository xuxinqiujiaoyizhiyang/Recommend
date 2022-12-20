import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from model.utils.Dataset import Dataset

# criteo数据集
class criteoDataSet(Dataset):
    def __init__(self, path, read_part=True, Sample_num=10000):
        super(criteoDataSet, self).__init__()
        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part:
            data_df = pd.read_csv(path, sep='\t', header=None, names=names, nrows=Sample_num)
        else:
            data_df = pd.read_csv(path, sep='\t', header=None, names=names)
        # 离散型变量和连续型变量
        sparse_feature = ['C' + str(i) for i in range(1, 27)]
        dense_feature = ['I' + str(i) for i in range(1, 14)]
        feature = sparse_feature + dense_feature

        # 缺失值填充
        data_df[sparse_feature] = data_df[sparse_feature].fillna('-1')
        data_df[dense_feature] = data_df[dense_feature].fillna(0)

        # 连续值分箱离散化操作
        est = KBinsDiscretizer(strategy='uniform', encode='ordinal', n_bins=100)
        data_df[dense_feature] = est.fit_transform(data_df[dense_feature])

        # 离散值进行编码
        data_df[feature] = OrdinalEncoder().fit_transform(data_df[feature])

        self.data = data_df[feature + ['label']].values