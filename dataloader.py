import os
import json
import numpy as np
import pandas as pd
import random

import torch.utils.data as util_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MyDataSet(Dataset):
    def __init__(self, sent, label=None):
        self.sent = sent
        self.label = label

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, idx):
        return {'sent': self.sent[idx], 'label': self.label[idx], 'index': idx}


def get_dataloader(data_dir, batch_size, train_ratio=0.8, seed=42, workers=4):
    # read data

    df_data = pd.read_csv(data_dir, sep='\t', header=None)
    df_data.columns = ['label', 'sent']
    sent = df_data.sent.values
    label = df_data.label.values

    X_train, X_test, y_train, y_test = train_test_split(sent, label, test_size=1-train_ratio, random_state=seed)
    
    train_dataset = MyDataSet(X_train, y_train)
    test_dataset = MyDataSet(X_test, y_test)

    train_loader = util_data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    test_loader = util_data.DataLoader(
        test_dataset, 
        batch_size=batch_size*10,
        shuffle=False,
        num_workers=workers,
    )

    return train_loader, test_loader


if __name__ == '__main__':
    from pprint import pprint
    train_loader, test_loader = get_dataloader(
        '003/data/sst2_shuffled.tsv', 
        4, train_ratio=0.8, seed=42, workers=4)
    
    for i, batch in enumerate(train_loader):
        pprint(batch)
        break
        
    for i, batch in enumerate(test_loader):
        pprint(batch)
        break

    
