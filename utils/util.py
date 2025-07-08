import json
import random
import os
import numpy as np
import torch
import torch.optim
import shutil
from tqdm import tqdm

from datetime import datetime
import os.path as osp
from model.ctrgcn import *
import math
from collections import Counter
import logging
import h5py
from model import tools
from torch.utils.data import Dataset, DataLoader,Subset,ConcatDataset,SubsetRandomSampler
from model.ctrgcn import *
from torch.utils.data import Dataset


def train_split(num_class, labels, n_labeled_per_class, n_unlabeled_per_class, dataset):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    if dataset == 'NTU':
        labels = np.argmax(labels, axis=1)
    elif dataset == 'UCLA' or 'K400':
        labels = labels -1

    for i in range(num_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)  
        
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])

        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs, train_unlabeled_idxs
    
class NTUDataset(Dataset):
    def __init__(self, args, x, y):
        self.args = args
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.args.dataset == 'NTU':
            label = np.argmax(self.y[index])
        elif self.args.dataset == 'K400':
            label = self.y[index]
        return [self.x[index], int(label)]
    
class NTUDataLoaders(object):
    def __init__(self, args,L_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, dataset ='NTU', case = 0, aug = 1, seg = 30):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.seg = seg
        self.args = args
        self.num_class= args.num_classes

        self.l_samples = L_SAMPLES_PER_CLASS
        self.u_samples = U_SAMPLES_PER_CLASS

        self.create_datasets()
        self.train_all_set = NTUDataset(self.args, self.train_X, self.train_Y)
        self.val_set = NTUDataset(self.args, self.val_X, self.val_Y)
        
        self.train_labeled_dataset = Subset(self.train_all_set, self.train_labeled_idxs)
        self.train_unlabeled_dataset = Subset(self.train_all_set, self.train_unlabeled_idxs)
        self.train_set = ConcatDataset([self.train_labeled_dataset, self.train_unlabeled_dataset])

        self.train_set_labels = [label for _, label in self.train_set]
        self.class_count=Counter(self.train_set_labels)

    def get_train_labeled_loader(self, batch_size, num_workers):

        num_samples = len(self.train_all_set)

        self.mask = np.ones(num_samples, dtype=int)
        self.mask[self.train_labeled_idxs] = 0     
        batch_size_labeled = batch_size
        while len(self.train_labeled_dataset) < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if len(self.train_labeled_dataset) < 16:
            batch_size_labeled = 16


        return DataLoader(self.train_labeled_dataset, batch_size=batch_size_labeled,
                            shuffle=True, num_workers=0,
                            collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=True), self.mask
    def get_train_loader(self, batch_size, num_workers):

        return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=True)

    def get_val_loader(self, batch_size, num_workers):
    
        return DataLoader(self.val_set, batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=True)



    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=0,
                          collate_fn=self.collate_fn_fix_test, pin_memory=True, drop_last=True)

    def get_train_labeled_size(self):
        return len(self.train_set_labels)
    
    def get_train_size(self):
        return len(self.train_Y)
    
    def get_class_weights(self):
        total_sum = sum(self.l_samples)

        class_weights = [x / total_sum for x in self.l_samples]
        return class_weights

    def get_class_counts_unlabeled(self):
        labels = []
        for _, label in self.train_unlabeled_dataset:
            labels.append(label)
        return Counter(labels)
    
    def get_class_counts_labeled(self):
        labels = []
        for _, label in self.train_labeled_dataset:
            labels.append(label)
        return Counter(labels)
    
    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        save_path = '.'
      
        if self.dataset == 'NTU':
            if self.args.configs == 'NTU60':
                if self.case ==0:
                    self.metric = 'CS'
                elif self.case == 1:
                    self.metric = 'CV'
            
            elif self.args.configs == 'NTU120':
                if self.case ==0:
                    self.metric = 'xset'
                elif self.case == 1:
                    self.metric = 'xsub'
            
            file_path = osp.join(save_path, 'ntu', self.args.configs, 'NTU_%s.h5' % (self.metric))

        elif self.dataset == 'K400':
          file_path = osp.join(save_path,'K400.h5')
        
        with h5py.File(file_path, 'r') as h5file:
            self.train_X = h5file['x'][:]
            self.train_Y = h5file['y'][:]
            
            self.val_X = h5file['valid_x'][:]
            self.val_Y = h5file['valid_y'][:]

        self.train_labeled_idxs, self.train_unlabeled_idxs = train_split(self.num_class, self.train_Y, self.l_samples, self.u_samples, self.dataset)

    def collate_fn_fix_train(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
    
        if self.dataset == 'K400':  
            x = np.array(x)
            x = torch.tensor(x) 
    
            x = x.permute(0,2,3,4,1)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2]* x.shape[3]*x.shape[4])
            x = np.array(x)
            
        x, y = self.Tolist_fix(x, y, train=1)
        
        if len(x) == 0:
            return None, None
        
        lens = np.array([x_.shape[0] for x_ in x], dtype=int)
        idx = lens.argsort()[::-1]  
        y = np.array(y)[idx]
        
        if len(x) == 1:
            x = torch.from_numpy(x[0]).unsqueeze(0)  
        else:
            x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
    
            if self.args.configs == 'NTU60':
                if self.case == 0:
                    theta = 0.3
                elif self.case == 1:
                    theta = 0.5
            elif self.args.configs == 'NTU120':
                theta = 0.3
            else:
                theta = 0.3
            x = _transform(x, theta, self.args.configs)
       
        y = torch.LongTensor(y)
    
        return [x, y]

    def collate_fn_fix_val(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, y = self.Tolist_fix(x, y, train=1)
        
        if len(x) == 0:
            return None, None
            
        idx = range(len(x))
        y = np.array(y)
    
        if len(x) == 1:
            x = torch.from_numpy(x[0]).unsqueeze(0)  
        else:
            x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
            
        y = torch.LongTensor(y)
    
        return [x, y]

    def collate_fn_fix_test(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=2)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        return [x, y]


    def Tolist_fix(self, joints, y, train = 1):
        seqs = []
        if self.dataset == 'NTU':
          for idx, seq in enumerate(joints):
            zero_row = []
            for i in range(len(seq)):
                if (seq[i, :] == np.zeros((1, 150))).all():
                        zero_row.append(i)

            seq = np.delete(seq, zero_row, axis = 0)

            seq = turn_two_to_one(seq)
            seqs = self.sub_seq(seqs, seq, train=train)

        elif self.dataset == 'K400':
          for idx, seq in enumerate(joints):
            zero_row = []
            
            for i in range(len(seq)):
                if (seq[i, :] == np.zeros((1, 108))).all():
                        zero_row.append(i)
            if len(zero_row) != len(seq):
                seq = np.delete(seq, zero_row, axis = 0)

                seq = turn_two_to_one(seq, self.dataset)

                seqs = self.sub_seq(seqs, seq, train=train)

           
        return seqs, y

    def sub_seq(self, seqs, seq , train = 1):
        group = self.seg

        if seq.shape[0] < self.seg:
            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            offsets1 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets2 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets3 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets4 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets5 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)

            seqs.append(seq[offsets1])
            seqs.append(seq[offsets2])
            seqs.append(seq[offsets3])
            seqs.append(seq[offsets4])
            seqs.append(seq[offsets5])

        return seqs

class K400Dataset(Dataset):
    def __init__(self, l_samples, u_samples, data_path='joint', mode='train', repeat=5, random_choose=True, random_shift=False, random_move=True, window_size=20, normalization=False, debug=False, use_mmap=True, root='kinetics\\kinetics-skeleton'):

        self.num_class = 400
        self.dataset = 'K400'
        self.l_samples = l_samples
        self.u_samples = u_samples

        self.root = root

        self.debug = debug
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.debug = debug
        if mode == 'train':
            self.data_path = os.path.join(root, 'kinetics_train')  
            self.label_path = os.path.join(root, 'kinetics_train_label.json')  
            self.label_pkl = os.path.join(root, 'train_label.json')
        else:
            self.data_path = os.path.join(root, 'kinetics_val')
            self.label_path = os.path.join(root, 'kinetics_val_label.json')
            self.label_pkl = os.path.join(root, 'val_label.json')
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = 5
        self.num_person_out = 2
        self.mmap = True
        self.ignore_empty_sample = True
        self.mode = mode

        self.load_data()

    def load_data(self):
        # data: N C V T M
        self.data = []

        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = []
        labels = []
        has_skeleton = []
        
        for name in tqdm(self.sample_name, desc="Processing samples"):
            id = name.split('.')[0]
            sample_id.append(id)
            
            try:
                label_idx = label_info[id]['label_index'] 
                has_skel = label_info[id]['has_skeleton']
                labels.append(label_idx)
                has_skeleton.append(has_skel)
            except KeyError:
                print(f"Warning: Missing label info for sample {id}")
                continue
                
        self.label = np.array(labels)
        has_skeleton = np.array(has_skeleton)

        if self.ignore_empty_sample:
            print("Filtering samples without skeleton...")
            self.sample_name = [
                s for h, s in zip(has_skeleton, self.sample_name) if h
            ]
            self.label = self.label[has_skeleton]

        self.N = len(self.sample_name) 
        self.C = 3  #channel
        self.T = 300  #frame
        self.V = 18  #joint
        self.M = self.num_person_out  #person

        if self.mode == "train":
            self.train_labeled_idxs, self.train_unlabeled_idxs = train_split(self.num_class, self.label, self.l_samples, self.u_samples, self.dataset)


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.sample_name)

  
    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)

        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        label = video_info['label_index']
        assert (self.label[index] == label)

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:  
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:    
            data_numpy = tools.random_move(data_numpy)

        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

class K400DataLoaders(object):
    def __init__(self, args,L_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, dataset ='NTU', case = 0, aug = 1, seg = 30):
        print("Initializing K400 data loaders...")

        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.seg = seg
        self.args = args
        self.num_class= args.num_classes

        self.l_samples = L_SAMPLES_PER_CLASS
        self.u_samples = U_SAMPLES_PER_CLASS
        root=args.dataroot + '/kinetics/kinetics-skeleton'
        with tqdm(total=2, desc="Loading datasets") as pbar:
            self.train_all_set = K400Dataset(mode='train', l_samples=self.l_samples, u_samples=self.u_samples, root=root)
            pbar.update(1)
            
            self.val_set = K400Dataset(mode='val', l_samples=self.l_samples, u_samples=self.u_samples, root=root)
            pbar.update(1)
        
        print("Processing data splits...")
        
        self.train_labeled_idxs = self.train_all_set.train_labeled_idxs
        self.train_unlabeled_idxs = self.train_all_set.train_unlabeled_idxs
        print("Creating data subsets...")

        self.train_labeled_dataset = Subset(self.train_all_set, self.train_labeled_idxs)
        self.train_unlabeled_dataset = Subset(self.train_all_set, self.train_unlabeled_idxs)
        self.train_set = ConcatDataset([self.train_labeled_dataset, self.train_unlabeled_dataset])

        self.train_set_labels = [label for _, label in self.train_set]
        self.class_count=Counter(self.train_set_labels)
        print("-" * 50)


    def get_train_labeled_loader(self, batch_size, num_workers=0):
        num_samples = len(self.train_all_set)
        self.mask = np.ones(num_samples, dtype=int)  
        self.mask[self.train_labeled_idxs] = 0  

        batch_size_labeled = batch_size
        while len(self.train_labeled_idxs) < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if len(self.train_labeled_idxs) < 16:
            batch_size_labeled = 16

        loader = DataLoader(
            self.train_all_set, 
            batch_size=batch_size_labeled,
            sampler=SubsetRandomSampler(self.train_labeled_idxs), 
            num_workers=num_workers,
            collate_fn=self.collate_fn_fix_train, 
            pin_memory=True, 
            drop_last=False
        )

        try:
            test_batch = next(iter(loader))
        except StopIteration:
            print("Warning: Data loader has no data")
        except Exception as e:
            print(f"Data loader test failed: {e}")

        return loader, self.mask


    def get_train_loader(self, batch_size, num_workers=0):
        return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=False)

    def get_val_loader(self, batch_size, num_workers=0):
    
        return DataLoader(self.val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_val, pin_memory=True, drop_last=False)



    def get_test_loader(self, batch_size, num_workers=0):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True, drop_last=True)

    def get_train_labeled_size(self):
        return len(self.train_labeled_dataset_labels)
    
    def get_train_size(self):
        return len(self.train_set_labels)
    
    def get_class_weights(self):
        total_sum = sum(self.l_samples)

        class_weights = [x / total_sum for x in self.l_samples]
        return class_weights

    def get_class_counts_unlabeled(self):
        labels = []
        for _, label in self.train_unlabeled_dataset:
            labels.append(label)
        return Counter(labels)
    
    def get_class_counts_labeled(self):
        labels = []
        for _, label in self.train_labeled_dataset:
            labels.append(label)
        return Counter(labels)
    
    def get_val_size(self):
        return len(self.val_set)

    def get_test_size(self):
        return len(self.test_Y)

    def collate_fn_fix_train(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)

        x = np.array(x)
        x = torch.tensor(x) 

        x = x.permute(0,2,3,4,1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]* x.shape[3]*x.shape[4])
        x = np.array(x)
            
        if len(x) == 0:
            return None, None
        
        lens = np.array([x_.shape[0] for x_ in x], dtype=int)
        idx = lens.argsort()[::-1]  
        y = np.array(y)[idx]
        
        if len(x) == 1:
            x = torch.from_numpy(x[0]).unsqueeze(0)  
        else:
            x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
            
            theta = 0.3 
            x = _transform(x, theta, self.args.configs)  
    
        y = torch.LongTensor(y)

        return [x, y]

    def collate_fn_fix_val(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x = np.array(x)
        x = torch.tensor(x) 

        x = x.permute(0,2,3,4,1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]* x.shape[3]*x.shape[4])
        x = np.array(x)

        if len(x) == 0:
            return None, None
        
        lens = np.array([x_.shape[0] for x_ in x], dtype=int)
        idx = lens.argsort()[::-1]  
        y = np.array(y)[idx]
        
        if len(x) == 1:
            x = torch.from_numpy(x[0]).unsqueeze(0)  
        else:
            x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
            
            theta = 0.3  
            x = _transform(x, theta, self.args.configs)  
    
        y = torch.LongTensor(y)
    
        return [x, y]

    def collate_fn_fix_test(self, batch):
        """Puts each data field into a tensor with outer dimension batch size
        """
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=2)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        return [x, y]

    def Tolist_fix(self, joints, y, train = 1):
        seqs = []
        
        for idx, seq in enumerate(joints):
            seqs.append(seq.squeeze(-1).reshape(-1))  
        
        if len(seqs) == 0:
            print(f"Warning: Tolist_fix returned empty seqs. joints shape: {[j.shape for j in joints]}")

    def sub_seq(self, seqs, seq , train = 1):
        group = self.seg

        if seq.shape[0] < self.seg:
            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            offsets1 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets2 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets3 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets4 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets5 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)

            seqs.append(seq[offsets1])
            seqs.append(seq[offsets2])
            seqs.append(seq[offsets3])
            seqs.append(seq[offsets4])
            seqs.append(seq[offsets5])

        return seqs
    
def turn_two_to_one(seq, dataset='NTU'):
    new_seq = list()
    if dataset == 'NTU':
        length = 75
    elif dataset == 'K400':
        length = 54
    elif dataset == 'UCLA':
        length = 52
    
    for idx, ske in enumerate(seq):
        if (ske[0:length] == np.zeros((1, length))).all():
            new_seq.append(ske[length:])
        elif (ske[length:] == np.zeros((1, length))).all():
            new_seq.append(ske[0:length])
        else:
            new_seq.append(ske[0:length])
            new_seq.append(ske[length:])
    
    return np.array(new_seq)

def _rot(rot):
    rot = rot.float()
    
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=2)

    ry1 = torch.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=2)

    rz1 = torch.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=2)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _transform(x, theta, configs):
    x = x.clone().detach().float() 
   
    x = x.contiguous().view(x.size()[:2] + (-1, 3)) 
        
    rot = x.new(x.size()[0], 3).uniform_(-theta, theta)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    
    rot = _rot(rot)
    
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)   
    x = torch.transpose(x, 2, 3)  
    x = x.contiguous().view(x.size()[:2] + (-1,)) 
    return x


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    return list(class_num_list)


def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def label_propagation(train_x_allq, train_y_allq, 
                      train_mask_allq, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.95, num_real_class=2, epsilon=0.1, tau_e=-9.5, **kwargs):   

    data_embed = torch.cat([train_x_allq[j] for j in range(len(train_x_allq))], 0)
    y_label = torch.cat([train_y_allq[j] for j in range(len(train_y_allq))], 0)
    mask_label = np.concatenate(train_mask_allq)
    
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps) 
    emb1 = torch.unsqueeze(emb_all, 1)  
    emb2 = torch.unsqueeze(emb_all, 0) 

    w = ((emb1 - emb2) ** 2).mean(2) 
    w = torch.exp(-w / 2)

    class_weights = torch.zeros(num_real_class).to(device)
    for i in range(num_real_class):
        class_weights[i] = torch.sqrt(1. / (y_label == i).sum().float()+np.finfo(float).eps)

    for i in range(n):
        if mask_label[i] == 0: 
            w[i] *= class_weights[int(y_label[i])]
            w[:, i] *= class_weights[int(y_label[i])]  

    topk, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32) 
    w = w * mask
    for i in range(num_real_class):
        class_weights[i] = torch.sqrt(1. / (y_label == i).sum().float()+np.finfo(float).eps)

    for i in range(n):
        if mask_label[i] == 0: 
            w[i] *= class_weights[int(y_label[i])]
            w[:, i] *= class_weights[int(y_label[i])] 
            
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    eigenvalues = torch.linalg.eigvals(s)
    cond_num = []
    
    y = torch.zeros(y_label.shape[0], num_real_class).to(device)  
    y.fill_(epsilon / (num_real_class - 1))  
    for i in range(n):
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1 - epsilon

    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y)  

    all_knn_label = torch.argmax(f, 1)  
    end_knn_label = f.cpu().numpy()

    class_counter = [0] * num_real_class
    
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            end_knn_label[i] = y_label[i].cpu()
        else:
            class_counter[all_knn_label[i]] += 1

    classwise_num = torch.zeros((num_real_class,)).to(device)

    for i in range(num_real_class):
        classwise_num[i] = class_counter[i] / max(class_counter)
    
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    energy_scores = -torch.logsumexp(f, dim=1)

    cpl_mask = energy_scores.le(tau_e* (classwise_num[max_idx] / (2. - classwise_num[max_idx])))  

    return end_knn_label, cpl_mask.cpu().numpy(), cond_num, eigenvalues

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def build_experts_in_on_model_uniform(args, configs=None, feq=False):
    classifier = Classifier(configs.hid_dim*16, args.num_classes)   
    classifier_freq = Classifier(configs.hid_dim*16, args.num_classes)   
    return classifier, classifier_freq

def build_classifier_scdnet(args, configs=None, feq=False):
    classifier = Classifier(configs.hid_dim*16, args.num_classes)   
    classifier_freq = Classifier(configs.hid_dim*16, args.num_classes)    
    return classifier, classifier_freq

def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)  
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train

def convert_coeff(x, eps=1e-6):
    x_freq = torch.fft.fft(x, dim=1)
    amp = torch.sqrt((x_freq.real + eps).pow(2) + (x_freq.imag + eps).pow(2))

    phase = torch.atan2(x_freq.imag, x_freq.real + eps)
    if amp.dim() == 2:
        stack_r = torch.stack((amp, phase), -1)
        stack_r = stack_r.permute(0, 2, 1)
    elif amp.dim() == 3:
        stack_r = torch.cat((amp, phase), dim=-1)
    return stack_r


def create_logger(args, log_pkg):
    """
    :param logger_file_path:
    :return:
    """
    current_time = datetime.now()
    timestamp_str = current_time.strftime('%Y_%m_%d_%H_%M_%S')
    
    if not os.path.exists(log_pkg):
        os.makedirs(log_pkg)
    log_filename = os.path.join(log_pkg, f'log_{timestamp_str}_{args.labeled_ratio}.log')

    logger = logging.getLogger()         
    logger.setLevel(logging.INFO)        

    file_handler = logging.FileHandler(log_filename)  
    console_handler = logging.StreamHandler()            


    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)     
    console_handler.setFormatter(formatter)    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def copy_files(files, destination_folder):

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    destination_folder_with_time = os.path.join(destination_folder, current_time)
    os.makedirs(destination_folder_with_time)
    
    for file_path in files:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            destination_path = os.path.join(destination_folder_with_time, file_name)
            shutil.copy(file_path, destination_path)
            print(f"Copied File: {file_path} to {destination_path}")
        else:
            print(f"warning: File {file_path} dose not exist, skipping copy.")


def compute_condition_number(d1, w, d2):
    s = d1 * w * d2
    try:
        U, S, Vh = torch.linalg.svd(s)
        condition_number = S[0] / S[-1]  
        return condition_number
    except:
        return float('inf') 

def calculate_confusion_matrix(pred_labels, true_labels, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(true_labels)):
        confusion_matrix[int(true_labels[i])][int(pred_labels[i])] += 1
    return confusion_matrix