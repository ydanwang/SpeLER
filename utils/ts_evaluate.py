
import torch
import torch.optim
import torch.fft as fft
from ts_utils import convert_coeff
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support, classification_report
    
def evaluate_multi_experts_uniform(args, val_loader, model, classifier=None, 
                                   classifier_feq=None, 
                                   loss=None, d=None, att=None, **kwargs):

    val_loss = 0
    val_accu = 0

    val_feq_loss = 0
    val_feq_accu = 0

    num_classes = args.num_classes
    sum_correct = 0
    reports = []

    preds_all = []
    true_all = []
    auroc_macro = []
    auroc_micro = []

    auprc = []

    all_labels = []

    sum_len = 0


    for i, (data, target) in enumerate(val_loader):
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        all_labels.append(target)

        preds = []
        preds_feq = []

        preds_all = []
        true_all = []

        with torch.no_grad():

            x_feq = fft.rfft(data, dim=-1)
            data_feq,_ = convert_coeff(x_feq)

            data = data.float() 
            data_feq = data_feq.float() 
            
    
            if args.use_cross_domain_gate:
                out = model(data, data_feq, cd=True)

            else:
                out = model(data, data_feq)

            feat_time =  out['feat_time']
            feat_feq =  out['feat_feq']

            val_pred_time = classifier(feat_time)
            val_pred_feq = classifier_feq(feat_feq)
            
            val_loss += loss(val_pred_time, target).item()
            val_feq_loss += loss(val_pred_feq, target).item()

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

            y_pred_prob = val_pred_all.detach().cpu()
            y = target.cpu()

            report = classification_report(y, pred_all.cpu(), output_dict=True)
            reports.append(report)

            test_labels_onehot = (F.one_hot(y.long(), num_classes)).numpy()

            try:
                auc_bs_micro = metrics.roc_auc_score(test_labels_onehot, y_pred_prob, average='micro')
            except:
                auc_bs_micro = float(0)
            try:
                auc_bs_macro = metrics.roc_auc_score(test_labels_onehot, y_pred_prob, average='macro')
            except:
                auc_bs_macro = float(0)

            auroc_macro.append(auc_bs_macro)
            auroc_micro.append(auc_bs_micro)

            auprc_bs = metrics.average_precision_score(test_labels_onehot, y_pred_prob)
            auprc.append(auprc_bs)

            val_accu += torch.sum(pred_time == target)
            val_feq_accu += torch.sum(pred_feq == target)
            sum_correct += torch.sum(pred_all == target)

            sum_len += len(target)


    total_accuracy_time = val_accu / sum_len
    total_accuracy_feq = val_feq_accu / sum_len

    total_acc = sum_correct.float() / sum_len

    precision, recall, f1_score, _ = precision_recall_fscore_support(true_all, preds_all, average='macro')

  
    avg_report = {}
    for key in reports[0].keys():
        if key != 'accuracy': 
            try:
                avg_report[key] = sum(report[key]['recall'] for report in reports) / len(reports)
            except KeyError:
                avg_report[key] = 0  


    if 'test' in kwargs:
        return {'per_class_accuracy_time': None, 'per_layer_accuracy_time':[None, None, None, None], \
                'per_class_accuracy_feq': None, 'per_layer_accuracy_feq': [None, None, None, None], \
                'total_accuracy_time': total_accuracy_time, 'total_accuracy_feq': total_accuracy_feq, 'total_acc': total_acc, 'per_class_accuracy_all': None, 'precision': precision, 'recall': recall, 'f1_score': f1_score, \
                'auprc': torch.mean(torch.tensor(auprc)),
                'macro_auroc': torch.mean(torch.tensor(auroc_macro)), 'micro_auroc': torch.mean(torch.tensor(auroc_micro)), 'avg_report': avg_report}

    else:
        return val_loss / sum_len, total_accuracy_time, total_accuracy_feq, total_acc, precision, recall, f1_score