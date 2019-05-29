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
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

writer = SummaryWriter('runs/exp18-tractogram_paddingfrenet-100epoch-2.1-ss20-gamma0.5-ft-dp_0.3')
blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train_files')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'slpaddingzero':
    dataset = sl_paddingzeroDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train_files')

    test_dataset = sl_paddingzeroDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'slpaddingrandom':
    dataset = sl_paddingrandomDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train_files')

    test_dataset = sl_paddingrandomDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)

elif opt.dataset_type == 'slpaddingfrenet':
    dataset = sl_paddingfrenetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train_files')

    test_dataset = sl_paddingfrenetDataset(
        root=opt.dataset,
        split='test_files',
        npoints=opt.num_points,
        data_augmentation=False)


elif opt.dataset_type == 'bspline':
    dataset = bsplineDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train_files')

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


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

'''
validationdataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size = opt.batchSize,
        shuffle = False,
        num_workers = int(opt.workers))
'''
print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

#print('test')

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
best_accuracy = 0

#print(num_batch)
for epoch in range(opt.nepoch):
    scheduler.step()
    #print(epoch)
    epoch_acc = torch.tensor([])
    epoch_loss = torch.tensor([])
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        target = target.squeeze()
        loss = F.nll_loss(pred, target)
        #writer.add_scalar('train/loss', loss.item(), i)
        epoch_loss = torch.cat((epoch_loss, torch.tensor([loss])), 0)
        if opt.feature_transform:
            loss += feature_transform_reguliarzer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        accuracy = correct.item() / float(opt.batchSize)
        epoch_acc = torch.cat((epoch_acc, torch.tensor([accuracy])), 0)
        #writer.add_scalar('train/accuracy', accuracy, i)
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

    if epoch % 1 == 0:
        #val_acc =torch.tensor([])
        val_loss = torch.tensor([])
        total_correct = 0
        total_testset = 0
        for j, data in enumerate(testdataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            target = target.squeeze()
            loss = F.nll_loss(pred, target)
            val_loss = torch.cat((val_loss, torch.tensor([loss])),0)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
            acc = correct.item()/float(opt.batchSize)
            #if acc > best_accuracy:
                #best_accuracy = acc
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), acc))
        
        val_acc = total_correct/float(total_testset)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            os.system('rm %s/*best_model*' % opt.outf)
            torch.save(classifier.state_dict(), '%s/cls_best_model_%d.pth' % (opt.outf, epoch))
        #print(len(testdataloader))
        #print(total_correct)
        #print(val_acc)
        #print(val_loss)
        writer.add_scalar('validation/loss', val_loss.mean(),epoch)
        writer.add_scalar('validation/accuracy',val_acc, epoch)

    writer.add_scalar('train/epoch_loss', epoch_loss.mean(), epoch)    
    writer.add_scalar('train/epoch_acc', epoch_acc.mean(), epoch)    


'''
total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    target = target.squeeze()
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    best_model_path = glob.glob('%s/*best_model*' % opt.outf)[0]
    classifier.load_state_dict(torch.load(best_model_path))
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]
    acc = total_correct / float(total_testset)
    #print(total_correct)
    #print(total_testset)

print(total_correct)
print(total_testset)
print("final accuracy {}".format(acc))
writer.add_scalar('test/accuracy',acc)
'''
