import pandas as pd
from model.utils.Dataset import Dataset
import numpy as np

class movieLens(Dataset):
    def __init__(self, path, read_part=True, sample_num=1000000, task='classification'):
        super(Dataset, self).__init__()
        dtype= {
            'userId': np.int32,
            'movieId': np.int32,
            'rating': np.float16
        }
        if read_part:
            data = pd.read_csv(path, sep=',', dtype=dtype, nrows=sample_num)
        else:
            data = pd.read_csv(path, sep=',', dtype=dtype)

        data = data.drop(columns=['timestamp'])

        if task == 'classification':
            data['rating'] = data.apply(lambda x:1 if x['rating'] > 3 else 0, axis=1).astype(np.int8)

        self.data = data.values