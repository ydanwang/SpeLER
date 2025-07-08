
import torch
import torch.optim
from utils.util import convert_coeff

def evaluate(args, val_loader, model, classifier=None, classifiers_of_each_layer=None, 
                                   classifier_feq=None, classifiers_feq_of_each_layer=None, 
                                   loss=None, d=None, att=None, **kwargs):

    val_loss = 0
    val_accu = 0

    val_feq_loss = 0
    val_feq_accu = 0

    sum_correct = 0

    preds_all = []
    true_all = []


    all_labels = []
    sum_len = 0

    for ind, (data, target) in enumerate(val_loader):
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        if data.size(0) == 1:
            print.info("Skipping batch with size 1 to avoid BatchNorm issues")
            continue
        all_labels.append(target)

        preds = []
        preds_feq = []

        preds_all = []
        true_all = []

        with torch.no_grad():

            data, target = data.float().cuda(), target.long().cuda()     
            data_feq = convert_coeff(data)
            data_feq = data_feq.float().cuda()
    
            out = model(data, data_feq)

            feat_time =  out['feat_time']
            feat_freq =  out['feat_feq']

            val_pred_time = classifier(feat_time, softmax=True)
            val_pred_feq = classifier_feq(feat_freq, softmax=True)
            
            val_loss += loss(val_pred_time.float(), target).item()
            val_feq_loss += loss(val_pred_feq.float(), target).item()

            val_pred_time = val_pred_time.data
            val_pred_feq = val_pred_feq.data

            pred_time = torch.argmax(val_pred_time, axis=1)
            preds.append(pred_time)

            pred_feq = torch.argmax(val_pred_feq, axis=1)
            preds_feq.append(pred_feq)

            val_pred_all = (val_pred_time + val_pred_feq) / 2.0

            pred_all = torch.argmax(val_pred_all, axis=1)

            preds_all.extend(pred_all.tolist())
            true_all.extend(target.tolist())

            val_accu += torch.sum(pred_time == target) 
            val_feq_accu += torch.sum(pred_feq == target)
            sum_correct += torch.sum(pred_all == target)

            sum_len += len(target)


    total_accuracy_time = val_accu / sum_len
    total_accuracy_feq = val_feq_accu / sum_len

    total_acc = sum_correct.float() / sum_len


    return val_loss / sum_len, total_accuracy_time, total_accuracy_feq, total_acc