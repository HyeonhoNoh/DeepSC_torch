from config import Config
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import torch

pd.set_option("max_colwidth", None)

class Dataset(Dataset):
    def __init__(self, config, train_or_test, transform=None, target_transform=None):
        if train_or_test == 'train':
            self.data = pd.read_csv(config.paths['data_path'] + "train_data.csv", sep='\t', header=None)
        elif train_or_test == 'test':
            self.data = pd.read_csv(config.paths['data_path'] + "test_data.csv", sep='\t', header=None)
        else:
            ValueError("Invalid value. Train or Test should be given.")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx].to_string(index=False, header=False)
        return data

    def batch_sample(self, batch_size):
        idx = torch.randint(0, len(self.data), [batch_size,1])
        data = [self.data.iloc[x].to_string(index=False, header=False) for x in idx]
        return data



class IterableDataset(IterableDataset):

    def __init__(self, train_or_test, transform=None):
        if train_or_test == 'train':
            self.data = pd.read_csv(config.paths['data_path'] + "train_data.csv", sep='\t', header=None, iterator=True, chunksize=1)
        elif train_or_test == 'test':
            self.data = pd.read_csv(config.paths['data_path'] + "test_data.csv", sep='\t', header=None, iterator=True, chunksize=1)
        else:
            ValueError("Invalid value. Train or Test should be given.")
        self.transform = transform

    def __iter__(self):
        for line in self.data:
            line = line['text'].item()
            yield line

if __name__ == '__main__':
    config = Config()
    ds = Dataset(config, 'train')
    print(ds.__len__())
    print(ds.__getitem__(0))

    ds_it = IterableDataset(config, 'train')
    print(next(ds_it))