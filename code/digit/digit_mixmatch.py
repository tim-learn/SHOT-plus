import argparse
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import loss
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import pdb
import math
import sys, copy
from tqdm import tqdm
import utils
import scipy.io as sio
import pickle
import copy
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import network
# from data_list import ImageList, ImageList_twice
from sklearn.metrics import confusion_matrix
from data_load import mnist, svhn, usps
from torchvision import transforms

import cv2
def _gaussian_blur(x, sigma=0.1):
    ksize = int(sigma + 0.5) * 8 + 1
    dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
    return torch.from_numpy(dst)

def split_target(args):
    train_bs = args.batch_size
    if args.dset == 's2m':
        train_target = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        train_target2 = mnist.MNIST_twice('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])) 
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
    elif args.dset == 'u2m':  
        train_target = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        train_target2 = mnist.MNIST_twice('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])) 
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))     
    elif args.dset == 'm2u':
        train_target = usps.USPS('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: _gaussian_blur(x, sigma=0.1)),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        train_target2 = usps.USPS_twice('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: _gaussian_blur(x, sigma=0.1)),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: _gaussian_blur(x, sigma=0.1)),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    dset_loaders = {}
    dset_loaders["target_te"] = DataLoader(test_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target2"] = DataLoader(train_target2, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)

    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.model == 'source':
        modelpath = args.output_dir + "/source_F.pt" 
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/source_B.pt"   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/source_C.pt"    
        netC.load_state_dict(torch.load(modelpath))
        pass
    else:
        modelpath = args.output_dir + "/target_F_" + args.savename + ".pt" 
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/target_B_" + args.savename + ".pt"   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/target_C_" + args.savename + ".pt"    
        netC.load_state_dict(torch.load(modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['target_te'])
        for i in range(len(dset_loaders['target_te'])):
            data = iter_test.next()
            # pdb.set_trace()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    top_pred, predict = torch.max(all_output, 1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100
    mean_ent = loss.Entropy(nn.Softmax(dim=1)(all_output))
    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.dset + '_test', 0, 0, acc, mean_ent.mean())
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n') 


    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['target'])
        for i in range(len(dset_loaders['target'])):
            data = iter_test.next()
            # pdb.set_trace()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    top_pred, predict = torch.max(all_output, 1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100
    mean_ent = loss.Entropy(nn.Softmax(dim=1)(all_output))

    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.dset + '_train', 0, 0, acc, mean_ent.mean())
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')     

    if args.ps == 0:
        est_p = (mean_ent<mean_ent.mean()).sum().item() / mean_ent.size(0)
        log_str = 'Task: {:.2f}'.format(est_p)
        print(log_str + '\n')
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        PS = est_p
    else:
        PS = args.ps

    if args.choice == "ent":
        value = mean_ent
    elif args.choice == "maxp":
        value = - top_pred
    elif args.choice == "marginp":
        pred, _ = torch.sort(all_output, 1)
        value = pred[:,1] - pred[:,0]
    else:
        value = torch.rand(len(mean_ent))

    predict = predict.numpy()
    train_idx = np.zeros(predict.shape)

    cls_k = args.class_num
    for c in range(cls_k):
        c_idx = np.where(predict==c)
        c_idx = c_idx[0]
        c_value = value[c_idx]

        _, idx_ = torch.sort(c_value)
        c_num = len(idx_)
        c_num_s = int(c_num * PS)
        # print(c, c_num, c_num_s) 

        for ei in range(0, c_num_s):
            ee = c_idx[idx_[ei]]
            train_idx[ee] = 1

    train_target.targets = predict
    new_src = copy.deepcopy(train_target)
    new_tar = copy.deepcopy(train_target2)

    # pdb.set_trace()

    if args.dset == 'm2u':

        new_src.train_data = np.delete(new_src.train_data, np.where(train_idx==0)[0], axis=0)
        new_src.train_labels = np.delete(new_src.train_labels, np.where(train_idx==0)[0], axis=0)

        new_tar.train_data = np.delete(new_tar.train_data, np.where(train_idx==1)[0], axis=0)
        new_tar.train_labels = np.delete(new_tar.train_labels, np.where(train_idx==1)[0], axis=0)

    else:

        new_src.data = np.delete(new_src.data, np.where(train_idx==0)[0], axis=0)
        new_src.targets = np.delete(new_src.targets, np.where(train_idx==0)[0], axis=0)

        new_tar.data = np.delete(new_tar.data, np.where(train_idx==1)[0], axis=0)
        new_tar.targets = np.delete(new_tar.targets, np.where(train_idx==1)[0], axis=0)

    # pdb.set_trace()

    return new_src, new_tar

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def data_load(args, txt_src, txt_tgt):

    if args.dset == 's2m': 
        train_target = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))    
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.dset == 'u2m':  
        train_target = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif args.dset == 'm2u':
        train_target = usps.USPS('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

    dset_loaders = {}
    dset_loaders["train"] = DataLoader(train_target, batch_size=args.batch_size*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=args.batch_size*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source"] = DataLoader(txt_src, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = DataLoader(txt_tgt, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.worker, drop_last=True)

    return dset_loaders

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, model, flag=True):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, predict, all_output, all_label, acc
    else:
        return accuracy*100, predict, all_output, all_label


def train(args, txt_src, txt_tgt):
    ## set pre-process
    dset_loaders = data_load(args, txt_src, txt_tgt)
    # pdb.set_trace()
    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    max_iter = args.max_epoch*max_len
    interval_iter = max_iter // 10

    if args.dset == 'u2m':
        netG = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netG = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netG = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.model == 'source':
        modelpath = args.output_dir + "/source_F.pt" 
        netG.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/source_B.pt"   
        netB.load_state_dict(torch.load(modelpath))
    else:
        modelpath = args.output_dir + "/target_F_" + args.savename + ".pt" 
        netG.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/target_B_" + args.savename + ".pt"   
        netB.load_state_dict(torch.load(modelpath))

    netF = nn.Sequential(netB, netC)
    optimizer_g = optim.SGD(netG.parameters(), lr = args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr = args.lr)

    base_network = nn.Sequential(netG, netF)
    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])

    list_acc = []
    best_ent = 100

    for iter_num in range(1, max_iter + 1):
        base_network.train()
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=max_iter)

        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source"])
            inputs_source, labels_source = source_loader_iter.next()
        try:
            inputs_target, _, target_idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, target_idx = target_loader_iter.next()
        
        targets_s = torch.zeros(args.batch_size, args.class_num).scatter_(1, labels_source.view(-1,1), 1)
        inputs_s = inputs_source.cuda()
        targets_s = targets_s.cuda()
        inputs_t = inputs_target[0].cuda()
        inputs_t2 = inputs_target[1].cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = base_network(inputs_t)
            outputs_u2 = base_network(inputs_t2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        ####################################################################
        all_inputs = torch.cat([inputs_s, inputs_t, inputs_t2], dim=0)
        all_targets = torch.cat([targets_s, targets_u, targets_u], dim=0)
        if args.alpha > 0:
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)
        else:
            l = 1
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, args.batch_size))
        mixed_input = utils.interleave(mixed_input, args.batch_size)  
        # s = [sa, sb, sc]
        # t1 = [t1a, t1b, t1c]
        # t2 = [t2a, t2b, t2c]
        # => s' = [sa, t1b, t2c]   t1' = [t1a, sb, t1c]   t2' = [t2a, t2b, sc]

        logits = base_network(mixed_input[0])
        logits = [logits]
        for input in mixed_input[1:]:
            temp = base_network(input)
            logits.append(temp)

        # put interleaved samples back
        # [i[:,0] for i in aa]
        logits = utils.interleave(logits, args.batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        train_criterion = utils.SemiLoss()

        Lx, Lu, w = train_criterion(logits_x, mixed_target[:args.batch_size], logits_u, mixed_target[args.batch_size:], 
            iter_num, max_iter, args.lambda_u)
        loss = Lx + w * Lu

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc, py, score, y = cal_acc(dset_loaders["train"], base_network, flag=False)
            mean_ent = torch.mean(Entropy(score))
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.dset + '_train', iter_num, max_iter, acc, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n') 


            acc, py, score, y = cal_acc(dset_loaders["test"], base_network, flag=False)
            mean_ent = torch.mean(Entropy(score))
            list_acc.append(acc)

            if best_ent > mean_ent:
                val_acc = acc
                best_ent = mean_ent
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.dset + '_test', iter_num, max_iter, acc, mean_ent)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')       

    idx = np.argmax(np.array(list_acc))
    max_acc = list_acc[idx]
    final_acc = list_acc[-1]

    log_str = '\n==========================================\n'
    log_str += '\nVal Acc = {:.2f}\nMax Acc = {:.2f}\nFin Acc = {:.2f}\n'.format(val_acc, max_acc, final_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()  

    # torch.save(base_network.state_dict(), osp.join(args.output_dir, args.log + ".pt"))
    # sio.savemat(osp.join(args.output_dir, args.log + ".mat"), {'y':best_y.cpu().numpy(), 
    #     'py':best_py.cpu().numpy(), 'score':best_score.cpu().numpy()})
    
    return base_network, py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mixmatch for Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_tar', type=str, default='ckps')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--dset', type=str, default='u2m', choices=['u2m', 'm2u','s2m'], help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")

    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--lambda_u', default=100, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--cls_par', type=float, default=0.1)
    parser.add_argument('--ssl', type=float, default=0.0) 
    parser.add_argument('--ps', type=float, default=0.0)
    parser.add_argument('--choice', type=str, default="ent", choices=["maxp", "ent", "marginp", "random"])
    parser.add_argument('--model', type=str, default="target", choices=["source", 'target'])
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--bottleneck', type=int, default=256)

    args = parser.parse_args()
    args.class_num = 10
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output_tar, 'seed' + str(args.seed), args.dset)
    args.savename = 'par_' + str(args.cls_par)
    if args.ssl > 0:
         args.savename += ('_ssl_' + str(args.ssl))
    if args.model == 'source':
        args.savename = "srconly"

    args.log = 'ps_' + str(args.ps) + '_' + args.savename
    args.mm_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.mm_dir):
        os.system('mkdir -p ' + args.mm_dir)
    if not osp.exists(args.mm_dir):
        os.mkdir(args.mm_dir)
    args.out_file = open(osp.join(args.mm_dir, "{:}_{:}.txt".format(args.log, args.choice)), "w")

    utils.print_args(args)
    txt_src, txt_tgt = split_target(args)
    train(args, txt_src, txt_tgt)
    
    args.out_file.close()