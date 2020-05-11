import os
import torch
from datetime import date

def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def get_cfg_value(value):
    if value[0] == '[' and value[-1] == ']':
        value = [get_cfg_value(v) for v in value[1:-1].split()]
        return value
    if value == 'y':
        return True
    if value == 'n':
        return False
    if is_int(value):
        return int(value)
    if is_float(value):
        return float(value)
    return value

def set_exp_name(cfg, modelname, dataname):
    exp = cfg['experiment_name']
    exp = exp.replace('DATE', str(date.today()))
    exp = exp.replace('MODEL', modelname.lower())
    exp += '_data-{}'.format(dataname.lower())
    cfg['experiment_name'] = exp
    return

def print_cfg(cfg, fileobj=None):
    for k in sorted(cfg.keys()):
        line = '%s : %s' % (k, cfg[k])
        if fileobj is None:
            print(line)
        else:
            fileobj.write(line + '\n')

def save_dict_to_file(dic, filename):
    f = open(filename, 'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(filename):
    f = open(filename, 'r')
    data = f.read()
    f.close()
    return eval(data)

def initialize_metrics():
    metrics = {}
    #metrics['acc'] = []
    #metrics['iou'] = []
    #metrics['prec'] = []
    #metrics['recall'] = []
    metrics['mse'] = []
    metrics['abse'] = []

    return metrics

def update_metrics(metrics, prediction, target):
    prediction = prediction.data.int().cpu()
    target = target.data.int().cpu()

    abs_err = torch.mean(torch.sum(abs(target-prediction)))
    mserr = torch.mean(torch.sum((target-prediction)**2))
    #correct = prediction.eq(target).sum().item()
    #acc = correct / float(target.size(0))

    #tp = torch.mul(prediction, target).sum().item() + 0.00001
    #fp = prediction.gt(target).sum().item()
    #fn = prediction.lt(target).sum().item()
    #tn = correct - tp

    #iou = float(tp) / (tp + fp + fn)
    #prec = float(tp) / (tp + fp)
    #recall = float(tp) / (tp + fn)

    metrics['abse'].append(abs_err)
    metrics['mse'].append(mserr)
    #metrics['prec'].append(prec)
    #metrics['recall'].append(recall)
    #metrics['acc'].append(acc)
    #metrics['iou'].append(iou)

def log_avg_metrics(writer, metrics, prefix, epoch):
    for k, v in metrics.items():
        if type(v) == list:
            v = torch.tensor(v)
        writer.add_scalar('%s/epoch_%s' % (prefix, k), v.mean().item(), epoch)

def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_nrom = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(x2_norm.transpose(-2, -1),
                        x1,
                        x2.transpose(-2, -1),
                        alpha=-2).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res 
