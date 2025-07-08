import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import queue
import time
import torch

import random
from model.ctrgcn import *

from torch.optim.lr_scheduler import MultiStepLR

from model.loss import sup_contrastive_loss_CoT

from model.ctrgcn_transformer import Model as CTRGCN
from utils.util import build_classifier_scdnet, set_seed, make_imb_data, \
    NTUDataLoaders,K400DataLoaders, convert_coeff, create_logger, copy_files, label_propagation

from utils.evaluate import evaluate
from arguments import parse_args
# from config_files.NTU60 import Config


args = parse_args()

vis_energy = False
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

wandb_use = False   

if wandb_use:
  import wandb
  wandb.login()

def main():
    if wandb_use:
      wandb.init(
            project="PE-Borderline",
        )
    exec(f'from config_files.{args.configs} import Config', globals())
    
    configs = Config()

    args.log_epoch = configs.log_epoch

    if args.epochs_unsupervised == 0 and args.epochs == 0:
        args.epochs_unsupervised = configs.epochs_unsupervised
        args.epochs = configs.epoch
        args.patience = configs.patience

    args.input_size = configs.input_channels
    args.seg = configs.seg
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_seed(args.random_seed)
    files_to_copy = [__file__, "utils/util.py", "model/ctrgcn.py"] 

    destination_folder = os.path.join(args.log_path, "saved_files_results", args.configs, 'im_'+str(args.imb_ratio_l)+'_label_'+str(args.labeled_ratio))

    args.destination_folder = destination_folder
    args.num_classes = configs.num_class

    copy_files(files_to_copy, destination_folder)
    logger = create_logger(args, destination_folder)
    
    if wandb_use:
        args.tau_e = wandb.config.tau_e

    logger.info('-' * 50)
    logger.info(__file__)
    for arg in vars(args):
        logger.info(f"Argument {arg}: {getattr(args, arg)}")

    # N: labeled, U: unlabeled

    L = 300  # Defined according to the labeled ratio
    N = 2700  # Defined according to the labeled ratio
    if args.dataset == 'NTU': 
        L_SAMPLES_PER_CLASS = make_imb_data(L, args.num_classes, args.imb_ratio_l)

        U_SAMPLES_PER_CLASS = make_imb_data(N, args.num_classes, args.imb_ratio_u)
 
    if args.dataset == 'K400': 
        L_SAMPLES_PER_CLASS = make_imb_data(L, args.num_classes, args.imb_ratio_l)

        U_SAMPLES_PER_CLASS = make_imb_data(N, args.num_classes, args.imb_ratio_u)

    # if args.dataset == 'UCLA': 
    #     L_SAMPLES_PER_CLASS = make_imb_data(124, args.num_classes, args.imb_ratio_l)

    #     U_SAMPLES_PER_CLASS = make_imb_data(124, args.num_classes, args.imb_ratio_u)

    total_sum = sum(L_SAMPLES_PER_CLASS)

    class_weights = [x / total_sum for x in L_SAMPLES_PER_CLASS]

    args.class_weights = class_weights

    args.out = args.dataset + '@N_' + str(1500) + '_r_'

    logger.info('-' * 50)

    model = CTRGCN(args.num_classes, in_channels=args.input_size, num_point=configs.num_point, num_person=configs.num_person, graph_args=dict(layout=configs.layout, mode='spatial'),hid_dim=configs.hid_dim).to(device)

    classifier, classifier_feq = build_classifier_scdnet(args, configs)
    projection_head = ProjectionHead(input_dim=configs.hid_dim*8).to(device) # hid_dim*16 for replicate the structure of the scd-net

    classifier = classifier.to(device)

    loss = nn.CrossEntropyLoss().to(device)

    projection_head_feq = ProjectionHead(input_dim=configs.hid_dim*8).to(device)   

    classifier_feq = classifier_feq.to(device)

    loss_feq = nn.CrossEntropyLoss().to(device)


    if args.dataset == 'NTU':
        data_loaders = NTUDataLoaders(args, L_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, args.dataset, args.case, seg=args.seg)
        train_loader = data_loaders.get_train_loader(args.batch_size, args.num_workers)
        val_loader = data_loaders.get_val_loader(args.batch_size, args.num_workers)
        
        train_labeled_loader, mask_train = data_loaders.get_train_labeled_loader(args.batch_size, args.num_workers)
        train_size = data_loaders.get_train_size()
        val_size = data_loaders.get_val_size()
    elif  args.dataset == 'K400':
        data_loaders = K400DataLoaders(args, L_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, args.dataset, args.case, seg=args.seg)
        train_loader = data_loaders.get_train_loader(args.batch_size, args.num_workers)
        val_loader = data_loaders.get_val_loader(args.batch_size, args.num_workers)
        
        train_labeled_loader, mask_train = data_loaders.get_train_labeled_loader(args.batch_size, args.num_workers)
        train_size = data_loaders.get_train_size()
        val_size = data_loaders.get_val_size()

    # elif args.dataset == 'UCLA':
    #     data_loaders = UCLADataLoaders(args, L_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS, args.dataset, args.case, seg=args.seg)
    #     train_loader = data_loaders.get_train_loader(args.batch_size, args.num_workers)
    #     val_loader = data_loaders.get_val_loader(args.batch_size, args.num_workers)
        
    #     train_labeled_loader, mask_train = data_loaders.get_train_labeled_loader(args.batch_size, args.num_workers)
    #     train_size = data_loaders.get_train_size()
    #     val_size = data_loaders.get_val_size()
        
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()) + list(projection_head.parameters()) + list(
            classifier_feq.parameters()) + list(projection_head_feq.parameters()),
            lr=args.lr,
            weight_decay=0.05,  
            betas=(0.9, 0.999)
    )
    if args.optimizer == 'ExponentialLR':

      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.optimizer == 'CosineAnnealingLR':

      scheduler = torch.optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=args.lr * 2,
          epochs=args.epochs,
          steps_per_epoch=len(train_loader),
          pct_start=0.1,  
          anneal_strategy='cos'
      )
    else:
      scheduler = MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

    logger.info('Training with optimizer {}'.format(args.optimizer))


    logger.info(f'Train on {train_size} samples, where {sum(L_SAMPLES_PER_CLASS)} samples are labeled, validate on {val_size} samples')


    val_loss = float('inf')

    best_val_acc = 0
    best_epoch = 0
    patience = args.patience  
    no_improve_epochs = 0
    best_model_states = None

    queue_train_x = queue.Queue(args.queue_maxsize)
    queue_train_y = queue.Queue(args.queue_maxsize)
    queue_train_mask = queue.Queue(args.queue_maxsize)

    queue_train_x_freq = queue.Queue(args.queue_maxsize)
    queue_train_y_feq = queue.Queue(args.queue_maxsize)

    for epoch in range(args.epochs):

        sum_correct = 0
        sum_len = 0
        num_iterations = 0
        correct = 0 
        total = 0 

        model.train()
        classifier.train()
        projection_head.train()

        classifier_feq.train()
        projection_head_feq.train()
                
        if epoch < args.epochs_unsupervised:

            for _, (x, y)  in enumerate(train_labeled_loader):
                if x is None:
                    continue
                    
                if x.size(0) == 1:
                    continue
                    
                x_freq = convert_coeff(x)

                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                x_freq = x_freq.to(device)

                out = model(x, x_freq)

                feat_time_5 = out['feat_time']
                feat_freq_5 = out['feat_feq']
                pred = classifier(feat_time_5)
                pred_feq = classifier_feq(feat_freq_5)

                val_pred_all = (pred + pred_feq) / 2.0
                
                pred_all = torch.argmax(val_pred_all, axis=1)
                sum_correct += torch.sum(pred_all == y)
                sum_len += len(y)

                step_loss = loss(pred.float(), y.long())
                step_loss_feq = loss_feq(pred_feq.float(), y.long())

                project_head_embed = projection_head(feat_time_5)
                project_head_embed_feq = projection_head_feq(feat_freq_5)

                if len(y) > 1:
                    batch_sup_contrastive_loss = sup_contrastive_loss_CoT(
                        embd_batch_1=project_head_embed, embd_batch_2=project_head_embed,
                        labels=y,
                        device=device,
                        temperature=0.05, 
                        base_temperature=20)

                    step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                    batch_sup_contrastive_loss_feq = sup_contrastive_loss_CoT(
                        embd_batch_1=project_head_embed_feq, embd_batch_2=project_head_embed_feq,
                        labels=y,
                        device=device,
                        temperature=0.05,
                        base_temperature=20)

                    step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_lambda

                all_loss = step_loss_feq + step_loss
                all_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()) + 
                    list(projection_head.parameters()) + list(classifier_feq.parameters()) + 
                    list(projection_head_feq.parameters()),
                    max_norm=1.0
                )

                optimizer.step()
                
                num_iterations = num_iterations + 1

        else:
            for ind, (x, y)  in enumerate(train_loader):
                if x.size(0) == 1:
                    logger.info("Skipping batch with size 1 to avoid BatchNorm issues")
                    continue
                x, y = x.to(device), y.to(device)
                x_freq = convert_coeff(x)
                x_freq = x_freq.to(device)
                target = y.clone()
             
                if (num_iterations + 1) * x.shape[0] < train_size:
                    mask_train_batch = mask_train[num_iterations * x.shape[0] : (num_iterations + 1) * x.shape[0]]  
                else:
                    mask_train_batch = mask_train[num_iterations * x.shape[0]:]

                optimizer.zero_grad()

                out = model(x, x_freq)

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

                        queue_train_x_freq.put(preject_head_embed_feq.detach())
                        queue_train_y_feq.put(y)

                    if queue_train_x.full():
    
                        train_x_allq = queue_train_x.queue
                        train_y_allq = queue_train_y.queue
                        train_mask_allq = queue_train_mask.queue

                        train_x_allq_feq = queue_train_x_freq.queue
                        train_y_allq_feq = queue_train_y_feq.queue


                        
                        end_knn_label, mask_cpl_knn, _, _ = label_propagation(
                            train_x_allq, train_y_allq,
                            train_mask_allq=train_mask_allq, device=device,
                            num_real_class=args.num_classes, topk=args.knn_num_tem, num_iterations=num_iterations,
                            logger=logger, tau_e=args.tau_e, alpha=args.alpha
                            )
                 
                        end_knn_label_feq, mask_cpl_knn_feq, _, _ = label_propagation(
                            train_x_allq_feq, train_y_allq_feq,
                            train_mask_allq=train_mask_allq, device=device,
                            num_real_class=args.num_classes, topk=args.knn_num_feq, num_iterations=num_iterations,
                            logger=logger, tau_e=args.tau_e, alpha=args.alpha)
                        

                        knn_result_label = torch.tensor(end_knn_label).to(device)
                        knn_result_label_feq = torch.tensor(end_knn_label_feq).to(device)

                        knn_result_fuse_label = (knn_result_label + knn_result_label_feq) / 2
                        pseudo_all = torch.argmax(knn_result_fuse_label, axis=1)
                        y = y.long()
                        y[mask_train_batch == 1] = pseudo_all[(len(pseudo_all) - len(y)):][mask_train_batch == 1]

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

                        _ = queue_train_x_freq.get()
                        _ = queue_train_y_feq.get()

                   
                        def calculate_metrics(confusion):
                            precision = np.diagonal(confusion) / (np.sum(confusion, axis=0) + 1e-6)
                            recall = np.diagonal(confusion) / (np.sum(confusion, axis=1) + 1e-6)
                            return precision, recall
                        
                mask_select_loss = [False for _ in range(len(y))]
                mask_select_loss_feq = [False for _ in range(len(y))]

                mask_pseudo_select = [False for _ in range(len(y))]
                for m in range(len(mask_train_batch)):
                    if mask_train_batch[m] == 0:
                        mask_select_loss[m] = True

                    else:
                        if mask_cpl_batch[m] or mask_cpl_batch_feq[m]:
                            mask_select_loss[m] = True
                            mask_pseudo_select[m] = True

                pred = classifier(feat_time_5)
                pred_feq = classifier_feq(feat_feq_5)

                val_pred_all = (pred + pred_feq) / 2.0
                
                pred_all = torch.argmax(val_pred_all, axis=1)
                sum_correct += torch.sum(pred_all == y)
                sum_len += len(y)

                step_loss = loss(pred[mask_select_loss].float(), y[mask_select_loss].long())
                step_loss_feq = loss_feq(pred_feq[mask_select_loss].float(), y[mask_select_loss].long())

                correct += torch.sum(y[mask_pseudo_select] == target[mask_pseudo_select])
                total += len(target[mask_pseudo_select])

                if len(y[mask_train_batch == 0]) > 1:
                    batch_sup_contrastive_loss = sup_contrastive_loss_CoT(
                        embd_batch_1=preject_head_embed[mask_train_batch == 0],
                        embd_batch_2=preject_head_embed[mask_train_batch == 0],
                        labels=y[mask_train_batch == 0],
                        device=device,
                        temperature=args.temperature,
                        base_temperature=args.temperature)

                    step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                    batch_sup_contrastive_loss_feq = sup_contrastive_loss_CoT(
                        embd_batch_1=preject_head_embed_feq[mask_train_batch == 0],
                        embd_batch_2=preject_head_embed_feq[mask_train_batch == 0],
                        labels=y[mask_train_batch == 0],
                        device=device,
                        temperature=args.temperature,
                        base_temperature=args.temperature)

                    batch_sup_contrastive_loss_cross = sup_contrastive_loss_CoT(
                        embd_batch_1=preject_head_embed[mask_train_batch == 0],
                        embd_batch_2=preject_head_embed_feq[mask_train_batch == 0],
                        labels=y[mask_train_batch == 0],
                        device=device,
                        temperature=args.temperature,
                        base_temperature=args.temperature)

                    step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_lambda + batch_sup_contrastive_loss_cross

                all_loss = step_loss_feq + step_loss
                all_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()) + 
                    list(projection_head.parameters()) + list(classifier_feq.parameters()) + 
                    list(projection_head_feq.parameters()),
                    max_norm=1.0
                )

                optimizer.step()
                num_iterations += 1

        scheduler.step()

        if sum_len > 0:
            total_acc = sum_correct / sum_len

        model.eval()
        classifier.eval()
        projection_head.eval()

        classifier_feq.eval()
        projection_head_feq.eval()
        
        start_time = time.time()

        val_loss, val_acc_time, val_acc_feq, val_acc = evaluate(
            args, val_loader, model, classifier=classifier, classifier_feq=classifier_feq, loss=loss)

        validation_duration = time.time() - start_time
        logger.info(f"Validation time: {validation_duration:.2f} seconds")
          
        if epoch > args.epochs_unsupervised:

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve_epochs = 0
                best_model_states = {
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'classifier_feq': classifier_feq.state_dict(),
                    'projection_head': projection_head.state_dict(),
                    'projection_head_feq': projection_head_feq.state_dict()
                }


                filename = f'best_model.pth'
                torch.save(best_model_states, os.path.join(destination_folder,filename))

                logger.info(f'New best model saved at epoch {epoch}, validation accuracy: {val_acc:.4f}, saved as {filename}')
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= patience:
                logger.info(f'Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with validation accuracy {best_val_acc:.4f}')
                model.load_state_dict(best_model_states['model'])
                classifier.load_state_dict(best_model_states['classifier'])
                classifier_feq.load_state_dict(best_model_states['classifier_feq'])
                projection_head.load_state_dict(best_model_states['projection_head'])
                projection_head_feq.load_state_dict(best_model_states['projection_head_feq'])
                break


        logger.info(
            "Epoch [{}/{}] Valid: \t Acc_tem : {:.5f} \t Acc_feq : {:.5f} \t Acc_all : {:.5f}"
            .format(epoch + 1, args.epochs, val_acc_time, val_acc_feq, val_acc))
        logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")
        if wandb_use:
          wandb.log(
              {
                "epoch": epoch,
                "train_acc": total_acc,
                "train_loss": all_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "val_acc": val_acc,
                "val_loss": val_loss,
                }
              )
            
if __name__ == '__main__':
        
    if wandb_use:
        sweep_configuration = {
            'method': 'random',  
            'metric': {'name': 'val_acc', 'goal': 'maximize'},              
            'parameters': {
                
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="")

        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()

