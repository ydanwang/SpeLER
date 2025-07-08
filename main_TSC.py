import os
import sys
from sklearn.model_selection import train_test_split

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import queue
import time
from collections import Counter
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import wandb
from data.tsc_data.dataset_load import *
from data.tsc_data.data_load import normalize_per_series, fill_nan_value
from model.ts_model.loss import sup_contrastive_loss_CoT
from model.ts_model.model import *
from utils.ts_utils import build_experts_in_on_model_uniform, set_seed, build_dataset,  \
    shuffler, convert_coeff, create_logger, copy_files, label_propagation

from utils.ts_evaluate import evaluate_multi_experts_uniform

use_wandb = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='classification', type=str, help='classification, regression')
    parser.add_argument('--note', default='crossdomaingate', type=str, help='experiment description')
    parser.add_argument('--use_cross_domain_gate', default=True, type=bool)
    
    parser.add_argument('--log_path', default='./results/', type=str, help='log file path to save result')

    # Base setup
    parser.add_argument('--backbone', type=str, default='FCN_expert_residual_gt_pymaid')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='HAR', 
                        help='dataset, imbalanced: SleepEDF, HAR, Epilepsy, SleepEEG, EMG') 
    parser.add_argument('--configs', type=str, default='HAR', help='config file: SleepEDF, HAR')

    parser.add_argument('--dataroot', type=str, default='E:/ts_data')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='0.1, 0.2, 0.5')
    
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3, 5 for ECG')
    parser.add_argument('--knn_num_tem', type=int, default=40, help='10, 20, 50')
    parser.add_argument('--knn_num_feq', type=int, default=30, help='10, 20, 50')
    parser.add_argument('--alpha', type=float, default=0.99, help='propagation factor')

    # Contrastive loss
    parser.add_argument('--sup_con_mu', type=float, default=0.3, help='weight for supervised contrastive loss: 0.05 or 0.005. exp:0.3')
    parser.add_argument('--sup_con_lambda', type=float, default=0.3, help='weight for pseudo contrastive loss: 0.05 or 0.005. exp:0.3')
    parser.add_argument('--tau_e', type=float, default=-9.5, help='')

    parser.add_argument('--temperature', type=float, default=50, help='20, 50')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function, cross_entropy, combined')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='128')

    parser.add_argument('--epochs', type=int, default=0, help='training epoch')
    parser.add_argument('--epochs_unsupervised', type=int, default=0, help='warmup epochs using only labeled data for ssl')
    parser.add_argument('--patience', type=int, default=0, help='training patience')

    parser.add_argument('--device', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    exec(f'from config_files.{args.configs}_Configs import Config as Configs')  
    configs = Configs()
    
    args.log_epoch = configs.log_epoch
    args.note = 'alpha='+str(args.alpha)

    if args.epochs_unsupervised == 0 and args.epochs == 0:
        
        args.epochs_unsupervised = configs.epochs_unsupervised
        args.epochs = configs.epoch
        args.patience = configs.patience

    args.input_size = configs.input_channels
    
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_seed(args.random_seed)

    if use_wandb:
        wandb.init(
        )
    if use_wandb:
        args.learning_rate = wandb.config.learning_rate
        args.tau_e = wandb.config.tau_e
        
    files_to_copy = [__file__, "ts_utils.py", "ts_model/model.py", "ts_model/loss.py", f"config_files/{args.configs}_Configs.py"] 
    destination_folder = os.path.join("saved_files_results", args.configs, args.dataset)
    args.destination_folder = destination_folder
    
    copy_files(files_to_copy, destination_folder)
    logger = create_logger(args, destination_folder)

    logger.info('-' * 50)
    logger.info(__file__)
    for arg in vars(args):
        logger.info(f"Argument {arg}: {getattr(args, arg)}")

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets, num_classes, num_class_list = build_dataset(args, logger)  
    
    args.num_classes = num_classes

    logger.info(f'classes distribution: {num_class_list}')
    logger.info('-' * 50)

    model_name = args.backbone
    model_function = getattr(sys.modules[__name__], model_name)
    model = model_function(num_classes, input_size=args.input_size, hid_dim=configs.hid_dim, device=device, configs=configs).to(device)

    classifier = build_experts_in_on_model_uniform(args, configs)
    projection_head = ProjectionHead(input_dim=configs.hid_dim)   

    classifier = classifier.to(device)
    projection_head = projection_head.to(device)

    loss = nn.CrossEntropyLoss().to(device)

    model_init_state = model.state_dict()
    classifier_init_state = classifier.state_dict()
    projection_head_init_state = projection_head.state_dict()

    classifier_feq = build_experts_in_on_model_uniform(args, configs, feq=True)

    projection_head_feq = ProjectionHead(input_dim=configs.hid_dim)

    classifier_feq = classifier_feq.to(device)

    projection_head_feq = projection_head_feq.to(device)

    loss_feq = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
                                list(model.parameters()) + list(classifier.parameters()) + list(projection_head.parameters()) + list(classifier_feq.parameters()) + list(projection_head_feq.parameters()) , 
                                lr=args.lr 
                            )       
    
    logger.info('on {}'.format(args.dataset))

    
    train_time = 0.0

    test_acc_avg_k_folds = []
    test_macro_auroc_k_folds = []
    test_micro_auroc_k_folds = []
    test_time = []

    test_auprc_k_folds = []
    for i, train_dataset in enumerate(train_datasets):
        if i > 0:
            break
        t = time.time()

        logger.info('{} fold start training and evaluate'.format(i))

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        train_dataset = normalize_per_series(train_dataset)
        val_dataset = normalize_per_series(val_dataset)
        test_dataset = normalize_per_series(test_dataset)

        if args.labeled_ratio == 1:
            train_all_split = train_dataset
            y_label_split = train_target

        else:
            train_labeled, train_unlabeled, y_labeled, y_unlabeled = train_test_split(train_dataset, train_target,
                                                                                    test_size= (1 - args.labeled_ratio),
                                                                                    random_state=args.random_seed)

            mask_labeled = np.zeros(len(y_labeled)) # [0, 0, ...., 0]
            mask_unlabeled = np.ones(len(y_unlabeled))  # [1, 1, ...., 1]
            mask_train = np.concatenate([mask_labeled, mask_unlabeled]) # [0, 0, ...., 1]

            train_all_split = np.concatenate([train_labeled, train_unlabeled])
            y_label_split = np.concatenate([y_labeled, y_unlabeled])

        x_train_all, y_train_all_org = shuffler(train_all_split, y_label_split)
        mask_train, _ = shuffler(mask_train, mask_train)   
        y_train_all = y_train_all_org.copy()   
        y_train_all[mask_train == 1] = -1  
        
        val_target_count = torch.tensor(val_target).long()
        label_counts = torch.bincount(val_target_count)
        class_weights = label_counts.float() / len(val_target_count)
        
        class_weights /= class_weights.sum()
        args.class_weights = class_weights

        class_counts = Counter(y_train_all)
        class_counts_list = [class_counts[i] for i in range(len(class_counts))]
        logger.info(f'Initial distribution of val labeled y :{class_counts_list}')

        train_fft = fft.rfft(torch.from_numpy(x_train_all), dim=-1)
        train_fft, _ = convert_coeff(train_fft)
        train_fft = train_fft.to(device)
        x_train_labeled_all_feq = train_fft[mask_train == 0]

        x_train_all = torch.from_numpy(x_train_all).to(device)
        y_train_all = torch.from_numpy(y_train_all).to(device).to(torch.int64)

        if x_train_all.dim() == 3:
            x_train_labeled_all = x_train_all[mask_train == 0] 
        elif x_train_all.dim() == 2:
            x_train_labeled_all = torch.unsqueeze(x_train_all[mask_train == 0], 1)  

        y_train_labeled_all = y_train_all[mask_train == 0]

        train_set_labled = Load_Dataset(x_train_labeled_all, y_train_labeled_all, configs)
        train_set = Load_Dataset(x_train_all, y_train_all, configs)   

        if isinstance(val_dataset, np.ndarray):
            val_dataset = torch.from_numpy(val_dataset)
            test_dataset = torch.from_numpy(test_dataset)

        val_set = Load_Dataset(val_dataset.to(device), torch.from_numpy(val_target).to(device).to(torch.int64), configs)
        test_set = Load_Dataset(test_dataset.to(device), torch.from_numpy(test_target).to(device).to(torch.int64), configs)
        
        batch_size_labeled = args.batch_size
        while x_train_labeled_all.shape[0] < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if x_train_labeled_all.shape[0] < 16:
            batch_size_labeled = 16

        train_labeled_loader = DataLoader(train_set_labled, batch_size=batch_size_labeled, num_workers=0,
                                        drop_last=False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        val_fft = fft.rfft(val_dataset, dim=-1)
        val_fft, _ = convert_coeff(val_fft)

        test_fft = fft.rfft(test_dataset, dim=-1)
        test_fft, _ = convert_coeff(test_fft)

        num_steps = args.epochs // args.batch_size

        stop_count = 0
        increase_count = 0
        real_eigenvalues_time = torch.tensor(0)
        real_eigenvalues_freq = torch.tensor(0)

        num_steps = train_set.__len__() // args.batch_size
        if num_steps == 0:
            num_steps = num_steps + 1

        val_loss = float('inf')
        test_accuracy = 0
        accuracy = 0


        queue_train_x = queue.Queue(args.queue_maxsize)
        queue_train_y = queue.Queue(args.queue_maxsize)
        queue_train_mask = queue.Queue(args.queue_maxsize)

        queue_train_x_feq = queue.Queue(args.queue_maxsize)
        queue_train_y_feq = queue.Queue(args.queue_maxsize)

        condition_numbers_time = []  
        condition_numbers_freq = [] 

        cond_num = 0
        cond_num_feq = 0
        for epoch in range(args.epochs):
            if stop_count == args.patience or increase_count == args.patience:
                logger.info('model convergent at epoch {}, early stopping'.format(epoch))
                break

            num_iterations = 0
            correct = 0 
            total = 0 
            model.train()
            classifier.train()
            projection_head.train()

            classifier_feq.train()
            projection_head_feq.train()

            if epoch < args.epochs_unsupervised:
                
                for ind, (x, y) in enumerate(train_labeled_loader):

                    if x.shape[0] < 2:
                        continue
                    if (num_iterations + 1) * batch_size_labeled < x_train_labeled_all_feq.shape[0]:
                        x_feq = x_train_labeled_all_feq[
                                num_iterations * batch_size_labeled: (num_iterations + 1) * batch_size_labeled]
                    else:
                        x_feq = x_train_labeled_all_feq[num_iterations * batch_size_labeled:]

                    optimizer.zero_grad()

                    if args.use_cross_domain_gate:
                        out = model(x, x_feq, cd=True)
                    else:
                        out = model(x, x_feq)

                    feat_time_5 = out['feat_time']
                    feat_feq_5 = out['feat_feq']
                    pred = classifier(feat_time_5)
                    pred_feq = classifier_feq(feat_feq_5)

                    step_loss = loss(pred, y)
                    step_loss_feq = loss_feq(pred_feq, y)


                    preject_head_embed = projection_head(feat_time_5)
                    preject_head_embed_feq = projection_head_feq(feat_feq_5)
                    if len(y) > 1:
                        batch_sup_contrastive_loss = sup_contrastive_loss_CoT(       
                            embd_batch_1=preject_head_embed,  embd_batch_2=preject_head_embed,
                            labels=y,
                            device=device,
                            temperature=0.05,        
                            base_temperature=20)

                        step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                        batch_sup_contrastive_loss_feq = sup_contrastive_loss_CoT(       
                            embd_batch_1=preject_head_embed_feq,  embd_batch_2=preject_head_embed_feq,
                            labels=y,
                            device=device,
                            temperature=0.05,
                            base_temperature=20)

                        step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_lambda

                        batch_sup_contrastive_loss_cross = sup_contrastive_loss_CoT(       
                            embd_batch_1=preject_head_embed,  embd_batch_2=preject_head_embed_feq,
                            labels=y,
                            device=device,
                            temperature=0.05,
                            base_temperature=20)
                    all_loss = step_loss_feq + step_loss + batch_sup_contrastive_loss_cross
                        
                    all_loss = step_loss_feq + step_loss 
                    all_loss.backward()

                    optimizer.step()

                    num_iterations = num_iterations + 1
            
            else:
                for ind, (x, y) in enumerate(train_loader):
                    if x.shape[0] < 2:
                        continue
                    if (num_iterations + 1) * args.batch_size < train_set.__len__():
                        x_feq = train_fft[
                                num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                        y_feq = y_train_all[
                                num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                        target = y_train_all_org[
                                num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]
                        mask_train_batch = mask_train[
                                        num_iterations * args.batch_size: (num_iterations + 1) * args.batch_size]    
                    else:
                        x_feq = train_fft[num_iterations * args.batch_size:]
                        y_feq = y_train_all[num_iterations * args.batch_size:]
                        mask_train_batch = mask_train[num_iterations * args.batch_size:]
                    
                    target = y.clone()

                    target = torch.tensor(target).to(device)    

                    optimizer.zero_grad()

                    if args.use_cross_domain_gate:
                        out = model(x, x_feq, cd=True)
                    else:
                        out = model(x, x_feq)

                    feat_time_5 = out['feat_time']
                    feat_feq_5 = out['feat_feq']

                    preject_head_embed = projection_head(feat_time_5)
                    preject_head_embed_feq = projection_head_feq(feat_feq_5)

                    mask_cpl_batch = torch.tensor([False] * len(mask_train_batch)).to(device)
                    mask_cpl_batch_feq = torch.tensor([False] * len(mask_train_batch)).to(device)

                    if epoch >= args.epochs_unsupervised:
                        if not queue_train_x.full():
                            queue_train_x.put(preject_head_embed.detach())
                            queue_train_y.put(y)
                            queue_train_mask.put(mask_train_batch)

                            queue_train_x_feq.put(preject_head_embed_feq.detach())
                            queue_train_y_feq.put(y_feq)

                        if queue_train_x.full():    
                            train_x_allq = queue_train_x.queue
                            train_y_allq = queue_train_y.queue
                            train_mask_allq = queue_train_mask.queue

                            train_x_allq_feq = queue_train_x_feq.queue
                            train_y_allq_feq = queue_train_y_feq.queue
                            
                            end_knn_label, mask_cpl_knn, cond_num = label_propagation(
                                train_x_allq, train_y_allq,
                                train_mask_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num_tem, 
                                num_iterations=num_iterations, logger=logger, tau_e=args.tau_e, alpha=args.alpha)
                            
                            end_knn_label_feq, mask_cpl_knn_feq, cond_num_feq = label_propagation(
                                train_x_allq_feq, train_y_allq_feq,
                                train_mask_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num_feq, 
                                num_iterations=num_iterations, logger=logger, tau_e=args.tau_e, alpha=args.alpha)


                            condition_numbers_time.append(cond_num)
                            condition_numbers_freq.append(cond_num_feq)

                            knn_result_label = torch.tensor(end_knn_label).to(device)
                            knn_result_label_feq = torch.tensor(end_knn_label_feq).to(device)

                            knn_result_fuse_label = (knn_result_label + knn_result_label_feq) / 2
                            pred_all = torch.argmax(knn_result_fuse_label, axis=1)

                                 
                            y[mask_train_batch == 1] = pred_all[(len(pred_all) - len(y)):][
                                                                mask_train_batch == 1] 

                            cpl_knn = mask_cpl_knn[(len(mask_cpl_knn) - len(y)):]
                            mask_train_batch = torch.tensor(mask_train_batch, device=device)
                            cpl_knn = torch.tensor(cpl_knn, device=mask_train_batch.device)

                            cpl_knn_all = cpl_knn[mask_train_batch == 1]
                            mask_cpl_batch[mask_train_batch == 1]  = cpl_knn_all.bool()

                            cpl_knn_feq = mask_cpl_knn_feq[(len(mask_cpl_knn_feq) - len(y)):]
                            cpl_knn_feq = torch.tensor(cpl_knn_feq, device=mask_train_batch.device)

                            cpl_knn_all_feq = cpl_knn_feq[mask_train_batch == 1]
                            mask_cpl_batch_feq[mask_train_batch == 1]  = cpl_knn_all_feq.bool()

                            _ = queue_train_x.get()
                            _ = queue_train_y.get()
                            _ = queue_train_mask.get()

                            _ = queue_train_x_feq.get()
                            _ = queue_train_y_feq.get()

                    mask_pseudo_select = [False for _ in range(len(y))]

                    mask_select_loss = [False for _ in range(len(y))]
                    mask_select_loss_feq = [False for _ in range(len(y))]
                    
                    for m in range(len(mask_train_batch)):
                        if mask_train_batch[m] == 0:
                            mask_select_loss[m] = True  
                                                    
                        else:
                            if mask_cpl_batch[m] or mask_cpl_batch_feq[m]:
                                mask_select_loss[m] = True
                        
                    pred = classifier(feat_time_5)
                    pred_feq = classifier_feq(feat_feq_5)

                    step_loss = loss(pred[mask_select_loss], y[mask_select_loss])
                    step_loss_feq = loss_feq(pred_feq[mask_select_loss], y[mask_select_loss])

                    correct += torch.sum(y[mask_pseudo_select] == target[mask_pseudo_select])
                    total += len(target[mask_pseudo_select])
                    
                    if len(y[mask_train_batch == 0]) > 1:
                        batch_sup_contrastive_loss = sup_contrastive_loss_CoT(      
                            embd_batch_1=preject_head_embed[mask_train_batch == 0], embd_batch_2=preject_head_embed[mask_train_batch == 0],
                            labels=y[mask_train_batch == 0],
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)

                        step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                        batch_sup_contrastive_loss_feq = sup_contrastive_loss_CoT(       
                            embd_batch_1=preject_head_embed_feq[mask_train_batch == 0], embd_batch_2=preject_head_embed_feq[mask_train_batch == 0],
                            labels=y_feq[mask_train_batch == 0],
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)
                        
                        batch_sup_contrastive_loss_cross = sup_contrastive_loss_CoT(      
                            embd_batch_1=preject_head_embed[mask_train_batch == 0], embd_batch_2=preject_head_embed_feq[mask_train_batch == 0],
                            labels=y_feq[mask_train_batch == 0],
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)

                        step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_lambda + batch_sup_contrastive_loss_cross

                    all_loss = step_loss_feq + step_loss

                    all_loss.backward()         

                    optimizer.step()
                    num_iterations += 1
                                
            if total > 0:
                pseudo_accuracy = correct / total
                logger.info(f'Accuracy of pseudo labels: {pseudo_accuracy * 100:.2f}%')

            model.eval()
            classifier.eval()
            projection_head.eval()

            classifier_feq.eval()
            projection_head_feq.eval()

            val_loss, val_acc_time, val_acc_feq, val_accu, precision, recall, f1_score = evaluate_multi_experts_uniform(args, val_loader, model, classifier=classifier, classifier_feq=classifier_feq, loss=loss)
            if epoch > args.epochs_unsupervised:
                if (abs(last_loss - val_loss) <= 1e-4):
                    stop_count += 1
                else:
                    stop_count = 0
            
                if (val_loss > last_loss):
                    increase_count += 1
                else:
                    increase_count = 0

            last_loss = val_loss

            logger.info(f"Epoch {epoch} condition number: Time {cond_num}, Freq: {cond_num_feq}")
            logger.info("Epoch [{}/{}] Valid: \t Acc_tem : {:.5f} \t Acc_feq : {:.5f} \t Acc_all : {:.5f}"
                            .format(epoch+1, args.epochs, val_acc_time, val_acc_feq, val_accu ))
            if use_wandb:
                wandb.log({
                        "epoch": epoch, 
                        f"acc_fold_{i+1}": val_accu, 
                        f"loss_fold_{i+1}": val_loss})
                

        t = time.time() - t
        train_time += t
        

        start_time = time.time()  
        out = evaluate_multi_experts_uniform(args, test_loader, model, classifier,  
                                                                classifier_feq,  
                                                                loss, test=True, d=f'mfsd_{args.dataset}_con', att=True)
     
        end_time = time.time()  
        inference_time = end_time - start_time
        logger.info(f"Final Testing time: {inference_time} s")

        test_accuracy = out['total_acc']

        macro_auroc = out['macro_auroc']
        micro_auroc = out['micro_auroc']
        auprc = out['auprc']

        avg_report = out['avg_report']

        test_macro_auroc_k_folds.append(macro_auroc)
        test_micro_auroc_k_folds.append(micro_auroc)
        test_time.append(inference_time)

        test_auprc_k_folds.append(auprc)

        test_acc_avg_k_folds.append(test_accuracy)

        logger.info(f'Test accuracy of {i}-th fold training: {test_accuracy}')

        test_per_class = out['per_class_accuracy_all']
        logger.info(f'Test accuracy of per class: {test_per_class}')

        logger.info("Test precision : {:.5f} \t Test recall : {:.5f} \t Test f1_score : {:.5f}".format(out['precision'], out['recall'], out['f1_score']))
        logger.info(f'Avg Report:{avg_report}')       
    
    test_acc_avg_k_folds = torch.Tensor(test_acc_avg_k_folds)
    mean_acc = torch.mean(test_acc_avg_k_folds).item()

    test_macro_auroc_avg_k_folds = torch.Tensor(test_macro_auroc_k_folds)
    test_micro_auroc_avg_k_folds = torch.Tensor(test_micro_auroc_k_folds)
    test_auprc_avg_k_folds = torch.Tensor(test_auprc_k_folds)
    test_time_avg_k_folds = torch.Tensor(test_time)
    
    mean_macro_auroc = torch.mean(test_macro_auroc_avg_k_folds).item()
    mean_micro_auroc = torch.mean(test_micro_auroc_avg_k_folds).item()
    mean_auprc = torch.mean(test_auprc_avg_k_folds).item()
    mean_test_time = torch.mean(test_time_avg_k_folds).item()

    logger.info(f"Traning Done: time (seconds) = {round(train_time, 3)}")

    logger.info(f"mean_macro_auroc = {mean_macro_auroc} ")
    logger.info(f"mean_micro_auroc = {mean_micro_auroc} ")
    logger.info(f'mean_auprc = {mean_auprc}')
    logger.info(f'mean_test_time = {mean_test_time}')

    test_per_class_accuracy_1, test_per_class_accuracy_2, test_per_class_accuracy_3, test_per_class_accuracy_4 = out['per_layer_accuracy_time']
    test_per_class_accuracy_5 = out['per_class_accuracy_time']
    test_per_class_accuracy_feq_1, test_per_class_accuracy_feq_2, test_per_class_accuracy_feq_3, test_per_class_accuracy_feq_4= out['per_layer_accuracy_feq']
    test_per_class_accuracy_feq_5 = out['per_class_accuracy_feq']

    logger.info(f"Mean of Test Accuracy = {mean_acc}, Traning Time (seconds) = {round(train_time, 3)}")
