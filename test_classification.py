from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import glob
from dataset import ModelNetDataset, bsplineDataset, sl_paddingzeroDataset, sl_paddingrandomDataset, sl_paddingfrenetDataset
from model import PointNetCls, feature_transform_reguliarzer
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, required=True, help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'modelnet40':

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'slpaddingzero':

    test_dataset = sl_paddingzeroDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'slpaddingrandom':

    test_dataset = sl_paddingrandomDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'slpaddingfrenet':

    test_dataset = sl_paddingfrenetDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)


elif opt.dataset_type == 'bspline':

    validation_dataset = bsplineDataset(
            root = opt.dataset,
            split = 'validation_files',
            npoints = opt.num_points,
            data_augmentation = False)
    
    test_dataset = bsplineDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))


# validationdataloader = torch.utils.data.DataLoader(
#         validation_dataset,
#         batch_size = opt.batchSize,
#         shuffle = False,
#         num_workers = int(opt.workers))

print(len(test_dataset))
num_classes = len(test_dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, input_size = 12, feature_transform=opt.feature_transform)

print('loading model %s' % opt.model)
if torch.cuda.is_available():
    classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

num_batch = len(test_dataset) / opt.batchSize

with torch.no_grad():
    #val_acc =torch.tensor([])
    test_loss = [torch.tensor([])]
    total_correct = 0
    total_testset = 0
    predictions = [torch.tensor([])]
    for j, data in enumerate(test_dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        pred, _, _ = classifier(points)
        target = target.squeeze()
        loss = F.nll_loss(pred, target)
        test_loss = torch.cat((test_loss, torch.tensor([loss])),0)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
        predictions = torch.cat((predictions, pred_choice), 0)
        acc = correct.item()/float(opt.batchSize)
        #if acc > best_accuracy:
            #best_accuracy = acc
        print('[%d/%d] accuracy: %f' % (j, num_batch, acc))
    
    test_acc = total_correct/float(total_testset)
    print('TEST - loss: %f accuracy: %f' % (test_loss.mean(), test_acc))

    np.save(opt.outf + '/prediction.npy', 
            predictions.cpu().numpy().astype(np.uint8))
