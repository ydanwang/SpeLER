import torch
import torch.nn as nn
import torch.nn.functional as F



EPS = 1e-8
        
def sup_contrastive_loss_CoT(embd_batch_1, embd_batch_2, labels, device, temperature=0.05, base_temperature=0.07):
    loss = 0
    cur_loss = 0
 
    anchor_dot_contrast = torch.div(   
        torch.matmul(embd_batch_1, embd_batch_2.T),  
        temperature)
     
    try:
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
    except:
        pass
        
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch_1.shape[0]).view(-1, 1).to(device), 
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    num_anchor = (mask.sum(1) != 0).sum().item()
            
    cur_loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = cur_loss.sum(0) /  (num_anchor + 1e-12)  
    return loss