from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import math, random
import h5py
from scipy.sparse import issparse

def loadIncMvSlDataFromMat(mat_path, fold_mat_path, fold_idx=0):
    # load incomplete multi-view multi-label data and labels
    # mark sure the out dimension is n x d, where n is the number of samples

    try:
        data = scipy.io.loadmat(mat_path)
        mv_data = data['X'][0]
    except Exception as e:
        print(str(e))
    datafold = scipy.io.loadmat(fold_mat_path)
    if 'Y' in data.keys():
        labels = data['Y']
    elif 'gt' in data.keys():
        labels = data['gt']
    elif 'truth' in data.keys():
        labels = data['truth']
    elif 'label' in data.keys():
        labels = data['label']
    else:
        raise ValueError('no label index key!!!', data.keys())
    labels = np.array(labels.astype(np.float32))
    if labels.min() == -1:
        labels = (labels + 1) * 0.5
    if labels.shape[0] in mv_data[0].shape:
        total_sample_num = labels.shape[0]
    elif labels.shape[1] in mv_data[0].shape:
        total_sample_num = labels.shape[1]
    if total_sample_num != mv_data[0].shape[0]:
        mv_data = [v_data.T for v_data in mv_data]
    if total_sample_num != labels.shape[0]:
        labels = labels.T
    for i, v_data in enumerate(mv_data):
        if issparse(v_data):
            mv_data[i] = v_data.toarray()
    ss_list = [StandardScaler() for i in range(len(mv_data))]
    mv_data = [ss_list[v].fit_transform(v_data.astype(np.float32)) for v, v_data in enumerate(mv_data)]
    # mv_data = [normalize(v_data.astype(np.float32)) for v_data in mv_data]
    folds_data = datafold['folds']  # (1,10)
    # folds_data = datafold['2']  # (1,10)
    max_folds_num = max(folds_data.shape)
    # folds_label = datafold['folds_label']
    # folds_sample_index = datafold['folds_sample_index']

    # incomplete data, label and random_sample index
    # inc_view_indicator = np.array(folds_data[0, fold_idx], 'int32')  # shape is [n x v]
    inc_view_indicator = np.array(folds_data[fold_idx], 'int32')  # shape is [n x v]
    # inc_view_indicator = np.array(folds_data[2], 'int32')

    # shuffle the data list
    random.seed(1)
    sample_index = list(range(total_sample_num))
    random.shuffle(sample_index)
    sample_index = np.array(sample_index)
    assert inc_view_indicator.shape[0] == sample_index.shape[0] == total_sample_num
    # incomplete data construction and normalization
    # inc_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))*inc_view_indicator[:,v:v+1])[sample_index,:] for v,v_data in enumerate(mv_data)]
    inc_mv_data = [(v_data.astype(np.float32)) * inc_view_indicator[:, v:v + 1] for v, v_data in enumerate(mv_data)]
    # v=1
    # v_data = mv_data[1]

    return [v_data[sample_index] for v_data in inc_mv_data], [v_data[sample_index] for v_data in mv_data], labels[
        sample_index], inc_view_indicator[sample_index], total_sample_num, ss_list


class ComDataset(Dataset):
    def __init__(self, mat_path, training_ratio=0.7, is_train=True, semisup=False):
        self.mv_data, self.labels, self.total_sample_num, self.ss_list = loadMvSlDataFromMat(mat_path)
        print(self.total_sample_num, training_ratio)
        self.train_sample_num = math.ceil(self.total_sample_num * training_ratio)
        self.test_sample_num = self.total_sample_num - self.train_sample_num
        if is_train:
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in self.mv_data]
            self.cur_labels = self.labels[:self.train_sample_num]

        else:
            self.cur_mv_data = [v_data[self.train_sample_num:] for v_data in self.mv_data]
            self.cur_labels = self.labels[self.train_sample_num:]
        # print('is_train:',is_train,'num:',self.cur_mv_data[0].shape)
        self.is_train = is_train
        self.classes_num = len(np.unique(self.labels))
        self.d_list = [da.shape[1] for da in self.mv_data]
        self.view_num = len(self.mv_data)

    def __len__(self):
        return self.train_sample_num if self.is_train else self.test_sample_num

    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index], dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        return data, label, torch.ones(len(data))


class IncDataset(Dataset):
    def __init__(self, mat_path, fold_mat_path, training_ratio=0.7, fold_idx=0, is_train=True, semisup=False):
        self.inc_mv_data, self.mv_data, self.labels, self.inc_V_ind, total_sample_num, self.ss_list = loadIncMvSlDataFromMat(
            mat_path, fold_mat_path, fold_idx)
        # inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num= loadMvMlDataFromMat(mat_path)
        self.train_sample_num = math.ceil(total_sample_num * training_ratio)
        self.test_sample_num = total_sample_num - self.train_sample_num
        if is_train:
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in self.inc_mv_data]
            self.cur_labels = self.labels[:self.train_sample_num]
            self.cur_inc_V_ind = self.inc_V_ind[:self.train_sample_num]
        else:
            self.cur_mv_data = [v_data[self.train_sample_num:] for v_data in self.inc_mv_data]
            self.cur_labels = self.labels[self.train_sample_num:]
            self.cur_inc_V_ind = self.inc_V_ind[self.train_sample_num:]

        self.is_train = is_train
        self.classes_num = len(np.unique(self.labels))
        self.d_list = [da.shape[1] for da in self.inc_mv_data]
        self.view_num = len(self.mv_data)
        # self.graph = self.__incGraph__()

    # def __incGraph__(self):
    #     graph = getMvKNNGraph(self.mv_data,k=15)
    #     graph = [v_graph.mul(self.inc_V_ind[:,v:v+1].mm(self.inc_V_ind[:,v:v+1].T)) for v,v_graph in enumerate(graph)]
    #     return graph
    def __len__(self):
        return self.train_sample_num if self.is_train else self.test_sample_num

    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index], dtype=torch.float) for v in self.cur_mv_data]
        Cdata = [torch.tensor(v[index], dtype=torch.float) for v in self.mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        return data, label, inc_V_ind, Cdata


def getComDataloader(matdata_path, training_ratio=1, is_train=True, batch_size=1, num_workers=1, shuffle=False):
    dataset = ComDataset(matdata_path, training_ratio=training_ratio, is_train=is_train)
    dataloder = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloder, dataset


def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=1, fold_idx=0, is_train=True, batch_size=1,
                     num_workers=1, shuffle=False):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, fold_idx=fold_idx,
                         is_train=is_train)
    return dataset
