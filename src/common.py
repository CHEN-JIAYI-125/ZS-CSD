import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def map_sequence(input_list):
    num_to_order = {}
    order = 0
    result = []
    for num in input_list:
        if num not in num_to_order:
            num_to_order[num] = order
            order += 1
        result.append(num_to_order[num])
    return result

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def compute_target_prototypes(H_lst, new_target, num_classes):
    # H_lst: List[List[tensor]], new_target: List[int]
    prototypes = []
    vectors = []
    labels = []
    for c in range(num_classes):
        prototype = []
        for i in range(len(new_target)):
            if new_target[i] == c:
                prototype.extend([utt for utt in H_lst[i]])
                vectors.extend([utt for utt in H_lst[i]])
                labels.extend([c for utt in H_lst[i]])
        if prototype:
            prototype = torch.stack(prototype).mean(dim=0)
        else:
            prototype = torch.zeros_like(H_lst[0][0])
        prototypes.append(prototype)
    return torch.stack(prototypes), torch.stack(vectors), torch.tensor(labels)

def target_CL(H_lst, target, config):
    # H_lst: List of utterance list; target: list of int/string
    tar_dic = {}
    tmp = 0
    for t in target:
        if t not in tar_dic:
            tar_dic[t] = tmp
            tmp += 1
    new_target = [tar_dic[t] for t in target]
    num_classes = tmp
    prototypes, vectors, labels = compute_target_prototypes(H_lst, new_target, num_classes)
    vectors = F.normalize(vectors, p=2, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    similarities = torch.mm(vectors, prototypes.T)
    similarities /= config.tau
    labels = labels.to(config.device)
    loss = F.cross_entropy(similarities, labels)
    return loss


def alllabel_supcon_loss(vectors, stance_labels, target_ids, tau=0.07, cross_target_weight=0.5):
    """
    PPED-style SupCon over all utterances in a batch.
    Positive: same stance; weight 1.0 if same target else cross_target_weight.
    """
    if vectors.size(0) <= 1:
        return vectors.new_zeros(())

    z = F.normalize(vectors, p=2, dim=-1)
    sim = torch.mm(z, z.t()) / max(tau, 1e-4)
    num = z.size(0)
    eye = torch.eye(num, device=vectors.device, dtype=torch.bool)

    same_stance = stance_labels.unsqueeze(0) == stance_labels.unsqueeze(1)
    same_target = target_ids.unsqueeze(0) == target_ids.unsqueeze(1)
    pos_mask = same_stance & ~eye
    pos_weight = torch.where(
        same_target,
        torch.ones_like(sim),
        torch.full_like(sim, cross_target_weight),
    )

    losses = []
    for i in range(num):
        pos_i = pos_mask[i]
        if not pos_i.any():
            continue
        exp_sim = torch.exp(sim[i]).masked_fill(eye[i], 0.0)
        denom = exp_sim.sum().clamp_min(1e-12)
        numer = (exp_sim * pos_weight[i] * pos_i.float()).sum().clamp_min(1e-12)
        losses.append(-torch.log(numer / denom))

    if not losses:
        return vectors.new_zeros(())
    return torch.stack(losses).mean()
