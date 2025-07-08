import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from collections import Counter
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler

def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes) 
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train

def load_data_UCR(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, 'UCR', dataset, dataset + '_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:] 
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, 'UCR', dataset, dataset + '_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    sum_dataset = pd.concat([train_x, test_x]).to_numpy(dtype=np.float32) 
    sum_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)

    num_classes = len(np.unique(sum_target))

    class_counts = Counter(sum_target)

    class_counts_list = [class_counts[i] for i in range(len(class_counts))]

    return sum_dataset, sum_target, num_classes, class_counts_list

def load_data_pt(dataroot, dataset, k_fold_flag=True):
    '''
    Dataset: ECG, EMG, FDA, FDB, Gesture, HAR, sleep-edf, SleepEEG
    '''
    train_data = torch.load(os.path.join(dataroot, dataset, 'train.pt'))
    val_data = torch.load(os.path.join(dataroot, dataset, 'val.pt'))
    test_data = torch.load(os.path.join(dataroot, dataset, 'test.pt'))

    if isinstance(train_data, list):
        train_x = train_data[0]
        train_target = train_data[1]
    else:
        train_x = train_data["samples"]
        train_target = train_data["labels"]

    if len(train_x.shape) < 3:
        train_x = train_x.unsqueeze(2)

    if isinstance(val_data, list):
        val_x = val_data[0]
        val_target = val_data[1]
    else:
        val_x = val_data["samples"]
        val_target = val_data["labels"]
    
    if len(val_x.shape) < 3:
        val_x = val_x.unsqueeze(2)

    if isinstance(test_data, list):
        test_x = test_data[0]
        test_target = test_data[1]
    else:
        test_x = test_data["samples"]
        test_target = test_data["labels"]
    
    if len(test_x.shape) < 3:
        test_x = test_x.unsqueeze(2)

    if train_x.shape[1] == 1:
        train_x_squeezed = train_x.squeeze(1)
        val_x_squeezed = val_x.squeeze(1)
        test_x_squeezed = test_x.squeeze(1)
    else:
        train_x_squeezed = train_x
        val_x_squeezed = val_x
        test_x_squeezed = test_x

    if isinstance(train_x_squeezed, torch.Tensor):
        sum_dataset = torch.concat([train_x_squeezed, val_x_squeezed, test_x_squeezed]).numpy().astype(np.float32) 
        sum_target = torch.concat([train_target, val_target, test_target]).numpy().astype(np.float32) 
    else:
        sum_dataset = np.concatenate([train_x_squeezed, val_x_squeezed, test_x_squeezed]).astype(np.float32) 
        sum_target = np.concatenate([train_target, val_target, test_target]).astype(np.float32) 
    
    x_train_all, y_train_all = shuffler(sum_dataset, sum_target)
    sum_dataset = torch.from_numpy(x_train_all)
    sum_target = torch.from_numpy(y_train_all).to(torch.int64)

    num_classes = len(np.unique(sum_target))

    class_counts = sum_target.bincount()
    
    if k_fold_flag:
        return sum_dataset, sum_target, num_classes, class_counts
    else:
        return train_x_squeezed, train_target, val_x_squeezed, val_target, test_x_squeezed, test_target, num_classes, class_counts



def transfer_labels(labels):

    unique_labels, new_indices = np.unique(labels, return_inverse=True)
    return new_indices


def k_fold(data, target, k=5):

    skf = StratifiedKFold(k, shuffle=True) 

    train_sets = []
    train_targets = []

    val_sets = []
    val_targets = []

    test_sets = []
    test_targets = []


    for raw_index, test_index in skf.split(data, target):  

        raw_set = data[raw_index]
        raw_target = target[raw_index]

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))

        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

        test_sets.append(data[test_index])
        test_targets.append(target[test_index])

    return train_sets, train_targets, val_sets, val_targets, test_sets, test_targets

def normalize_per_series(data):
    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    return (data - data.mean(axis=1, keepdims=True)) / std_

def normalize_uea_set(data_set):
    '''
    The function is the same as normalize_per_series, but can be used for multiple variables.
    '''
    return TimeSeriesScalerMeanVariance().fit_transform(data_set)

def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    numpy_array = np.take(col_mean, ind[1])

    tensor_to_assign = torch.from_numpy(numpy_array)

    train_set[ind] = tensor_to_assign

    ind_val = np.where(np.isnan(val_set))

    val_set_array= np.take(col_mean, ind_val[1])
    val_to_assign = torch.from_numpy(val_set_array)
    
    val_set[ind_val]= val_to_assign

    ind_test = np.where(np.isnan(test_set))
    test_set_array = np.take(col_mean, ind_test[1])
    test_to_assign = torch.from_numpy(test_set_array)

    test_set[ind_test] = test_to_assign

    return train_set, val_set, test_set

def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list