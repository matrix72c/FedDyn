import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def get_ncf_data(train_data_path, test_data_path, neg_pos_ratio):
    """
    Load data from files, and convert them into [[user_id, item_id], 0/1] format.
    """
    train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    pos_num = len(train_data)
    train_data = train_data.values.tolist()
    train_label = [1 for _ in range(pos_num)]

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    neg_ui = []
    for user, item in train_data:
        train_mat[user, item] = 1.0
        for t in range(neg_pos_ratio):
            j = np.random.randint(item_num)
            while (user, j) in train_mat:
                j = np.random.randint(item_num)
            neg_ui.append([user, j])
    train_data += neg_ui
    train_label += [0 for _ in range(pos_num * neg_pos_ratio)]

    with open(test_data_path, "r") as f:
        lines = f.readlines()
    test_list = [[] for _ in range(len(lines))]
    ret_test = []
    for line in lines:
        tmp_str_list = line.split("\t")
        user_str, item_str = tmp_str_list[0][1:-1].split(",")
        item_list = [int(i) for i in tmp_str_list[1:]]
        user = int(user_str)
        gt_item = int(item_str)
        item_list.append(gt_item)
        test_list[user] = [item_list, gt_item]
    for i in range(len(test_list)):
        tmp = []
        for j in range(len(test_list[i][0])):
            tmp.append([i, test_list[i][0][j]])
        ret_test.append([tmp, test_list[i][1]])
    return user_num, item_num, train_data, train_label, ret_test


def distribute_data(train_data, train_label, user_num):
    """
    Distribute the data into each user client(each user has an independent client).
    """
    clients_train_data, clients_train_label = [[] for _ in range(user_num)], [[] for _ in range(user_num)]
    for i in range(len(train_data)):
        user, item = train_data[i][0], train_data[i][1]
        clients_train_data[user].append([user, item])
        clients_train_label[user].append(train_label[i])
    return clients_train_data, clients_train_label

def torch_delete(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

class NCFDataset(Dataset):
    """
    NCF dataset for PyTorch DataLoader
    """

    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)