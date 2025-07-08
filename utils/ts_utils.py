import random
import os
import numpy as np
import torch
import shutil
import sys
from datetime import datetime
from data.tsc_data.data_load import k_fold
from data.tsc_data.data_load import *
from model.ts_model.model import *
import torch
import logging


import torch
import numpy as np


def save_results(destination_folder, acc, f1):
    run_metrics = {'accuracy': acc, 'f1_score': f1}
    metrics = {'accuracy': [], 'f1_score': []}

    df = pd.DataFrame(columns=["acc", "f1"])
    df.loc[0] = [acc, f1]

    for (key, val) in run_metrics.items(): 
        metrics[key].append(val)

    scores_save_path = os.path.join(destination_folder, "scores.xlsx")
    df.to_excel(scores_save_path, index=False)
    
def build_adjacency_matrix(data_embed, y_label, mask_label, num_real_class, device, sigma=0.25, topk=5):

    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d

    adjacency = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    adjacency = torch.exp(-adjacency / 2)

    class_weights = torch.zeros(num_real_class).to(device)
    for i in range(num_real_class):
        class_weights[i] = 1. / (y_label == i).sum().float()

    for i in range(n):
        if mask_label[i] == 0:  
            adjacency[i] *= class_weights[int(y_label[i])]

    topk, indices = torch.topk(adjacency, topk)
    mask = torch.zeros_like(adjacency).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  
    adjacency = adjacency * mask

    d = adjacency.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    normalized_adjacency = d1 * adjacency * d2

    return normalized_adjacency

def compute_energy_score(logits, T=1):
    exp_logits = torch.exp(logits / T)
    
    sum_exp_logits = torch.sum(exp_logits, dim=1)
    
    energy_scores = -T * torch.log(sum_exp_logits)
    
    return energy_scores

def compute_condition_number(d1, w, d2):
   
    s = d1 * w * d2
    try:
        U, S, Vh = torch.linalg.svd(s)
        condition_number = S[0] / S[-1]  
        return condition_number
    except:
        return float('inf')  

def label_propagation(train_x_allq, train_y_allq, train_mask_allq, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.95, num_real_class=2, epsilon=0.1, tau_e=-9.5, **kwargs):

    data_embed = torch.cat([train_x_allq[j] for j in range(len(train_x_allq))], 0)
    y_label = torch.cat([train_y_allq[j] for j in range(len(train_y_allq))], 0)
    mask_label = np.concatenate(train_mask_allq)
    
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d

    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    class_weights = torch.zeros(num_real_class).to(device)
    topk, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  
    w = w * mask
    for i in range(num_real_class):
        class_weights[i] = torch.sqrt(1. / (y_label == i).sum().float())

    for i in range(n):
        if mask_label[i] == 0: 
            w[i] *= class_weights[int(y_label[i])]
            w[:, i] *= class_weights[int(y_label[i])] 
            
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2
    cond_num = compute_condition_number(d1, w, d2)
    y = torch.zeros(y_label.shape[0], num_real_class).to(device)    
    y.fill_(epsilon / (num_real_class - 1))
    for i in range(n):
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1 - epsilon
    gamma = 0.75   
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y)  
    f = f * gamma  
    energy_scores = -torch.logsumexp(f, dim=1)

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

    threshold = tau_e* (classwise_num[max_idx] / (2. - classwise_num[max_idx]))
    cpl_mask = energy_scores.le(threshold)    

    return end_knn_label, cpl_mask, cond_num

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def build_dataset(args, logger, k_fold_flag=True): 

    function_name = f"load_{args.dataset}"

    try:
            
        if args.dataset in [
            'ECG',
            'EMG',
            'FDA',
            'FDB',
            'Gesture',
            'HAR',
            'SleepEDF',
            'SleepEEG',
            'Gesture',
            'Epilepsy',
        ]:
            if k_fold_flag:
                sum_dataset, sum_target, num_classes, num_class_list = load_data_pt(args.dataroot, args.dataset)
                sum_target = transfer_labels(sum_target)
            else:
                train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets, num_classes, num_class_list = load_data_pt(args.dataroot, args.dataset, k_fold_flag=False)
        
        else:                  
            loader_function = getattr(sys.modules[__name__], function_name)
            
            if k_fold_flag:
                sum_dataset, sum_target, num_classes, num_class_list = loader_function(args.dataroot, args.dataset)
                sum_target = transfer_labels(sum_target)
            else:
                train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets, num_classes, num_class_list = loader_function(args.dataroot, args.dataset, k_fold_flag=False)
        
        if k_fold_flag:
            args.seq_len = sum_dataset.shape[1]
    
            while sum_dataset.shape[0] * 0.6 < args.batch_size:
                args.batch_size = args.batch_size // 2

            if args.batch_size * 2 > sum_dataset.shape[0] * 0.6:
                logger.info('queue_maxsize is changed to 2')
                args.queue_maxsize = 2

            if k_fold_flag:
                train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = k_fold(
                    sum_dataset, sum_target)   
                
        return train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets, num_classes, num_class_list
    
    except AttributeError as e:
        print(f"An AttributeError occurred: {e}")
        raise ValueError(f"Can't find the dataset loader function named '{function_name}'.")

def build_experts_in_on_model_uniform(args, configs=None, feq=False):

    classifier = Classifier(configs.hid_dim, args.num_classes)    
    return  classifier



def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)  
    y_train = y_train[indexes]
    return x_train, y_train

def get_all_datasets(data, target):
    return k_fold(data, target)

def convert_coeff(x, eps=1e-6):

    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))

    phase = torch.atan2(x.imag, x.real + eps)

    if amp.dim() == 2:
        stack_r = torch.stack((amp, phase), -1)
        stack_r = stack_r.permute(0, 2, 1)
    elif amp.dim() == 3:
        stack_r = torch.cat((amp, phase), dim=1)  
    return stack_r, phase

import torch
import pywt

def convert_wavelet(x, wavelet='db1', level=1):
   
    if x.dim() == 2:
        x_np = x.cpu().numpy()
        coeffs = pywt.wavedec(x_np, wavelet=wavelet, level=level, axis=-1)
        coeffs_high = coeffs[-level:]
        coeffs_low = coeffs[0]
        
        coeffs_high_tensor = [torch.tensor(c, dtype=x.dtype) for c in coeffs_high]
        coeffs_low_tensor = torch.tensor(coeffs_low, dtype=x.dtype)
        
        return coeffs_high_tensor, coeffs_low_tensor
    
    elif x.dim() == 3:
        batch_size, channels, signal_length = x.shape
        coeffs_high = []
        coeffs_low = []
        for b in range(batch_size):
            batch_coeffs_high = []
            batch_coeffs_low = []
            for c in range(channels):
                signal = x[b, c].cpu().numpy()
                coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, axis=-1)
                batch_coeffs_high.append(coeffs[-level:])
                batch_coeffs_low.append(coeffs[0])
            coeffs_high.append(batch_coeffs_high)
            coeffs_low.append(batch_coeffs_low)
        
        coeffs_high_tensor = [
            [torch.tensor(c, dtype=x.dtype) for c in batch]
            for batch in coeffs_high
        ]
        coeffs_low_tensor = torch.tensor(coeffs_low, dtype=x.dtype)
        
        return coeffs_high_tensor, coeffs_low_tensor
    
    else:
        raise ValueError("The dimension of the input tensor must be 2 or 3.")
    

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