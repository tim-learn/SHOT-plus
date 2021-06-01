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
from data_list import ImageList, ImageList_twice
from sklearn.metrics import confusion_matrix

def split_target(args):
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_tar = open(args.t_dset_path).readlines()
    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()

    dset_loaders = {}
    test_set = ImageList(txt_tar, transform=test_transform)
    dset_loaders["target"] = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.model == "source":
        modelpath = args.output_dir + "/source_F.pt" 
        netF.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/source_B.pt"   
        netB.load_state_dict(torch.load(modelpath))
        modelpath = args.output_dir + "/source_C.pt"    
        netC.load_state_dict(torch.load(modelpath))
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
        iter_test = iter(dset_loaders['target'])
        for i in range(len(dset_loaders['target'])):
            data = iter_test.next()
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

    if args.dset == 'VISDA-C':
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        all_acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        acc = all_acc.mean()
        aa = [str(np.round(i, 2)) for i in all_acc]
        acc_list = ' '.join(aa)
        print(acc_list)
        args.out_file.write(acc_list + '\n')
        args.out_file.flush()

    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, 0, 0, acc, mean_ent.mean())
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

    ori_target = txt_tar.copy()
    new_tar = []
    new_src = []

    predict = predict.numpy()

    cls_k = args.class_num
    for c in range(cls_k):
        c_idx = np.where(predict==c)
        c_idx = c_idx[0]
        c_value = value[c_idx]

        _, idx_ = torch.sort(c_value)
        c_num = len(idx_)
        c_num_s = int(c_num * PS)
        
        for ei in range(0, c_num_s):
            ee = c_idx[idx_[ei]]
            reci = ori_target[ee].strip().split(' ')
            line = reci[0] + ' ' + str(c) + '\n' 
            new_src.append(line)
        for ei in range(c_num_s, c_num):
            ee = c_idx[idx_[ei]]
            reci = ori_target[ee].strip().split(' ')
            line = reci[0] + ' ' + str(c) + '\n' 
            new_tar.append(line)

    return new_src.copy(), new_tar.copy()

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def data_load(args, txt_src, txt_tgt):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dsets = {}
    dsets["source"] = ImageList(txt_src, transform=train_transform)
    dsets["target"] = ImageList_twice(txt_tgt, transform=[train_transform, train_transform])

    txt_test = open(args.test_dset_path).readlines()
    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()
    dsets["test"] = ImageList(txt_test, transform=test_transform)

    dset_loaders = {}
    dset_loaders["source"] = torch.utils.data.DataLoader(dsets["source"], batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = torch.utils.data.DataLoader(dsets["target"], batch_size=args.batch_size,
        shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=args.batch_size*3,
        shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, model):
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

    return accuracy*100, predict, all_output, all_label


def train(args, txt_src, txt_tgt):
    ## set pre-process
    dset_loaders = data_load(args, txt_src, txt_tgt)

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    max_iter = args.max_epoch*max_len
    interval_iter = max_iter // 10

    if args.net[0:3] == 'res':
        netG = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netG = network.VGGBase(vgg_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netG.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.model == "source":
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
            acc, py, score, y = cal_acc(dset_loaders["test"], base_network)
            mean_ent = torch.mean(Entropy(score))
            
            list_acc.append(acc)

            if best_ent > mean_ent:
                val_acc = acc
                best_ent = mean_ent
                best_y = y
                best_py = py
                best_score = score

            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%; Mean Ent = {:.4f}'.format(args.name, iter_num, max_iter, acc, mean_ent)
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
    
    return base_network, py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mixmatch for Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_tar', type=str, default='ckps')
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=36, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--bottleneck_dim', type=int, default=256)

    parser.add_argument('--net', type=str, default='resnet50', choices=["resnet18", "resnet50", "resnet101", "resnet34", "vgg16"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['pacs', 'DomainNet', 'DomainNet126', 'VISDA-C', 'office', 'office-home', 'image-clef'], help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")

    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=100, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--ssl', type=float, default=0.0) 
    parser.add_argument('--ps', type=float, default=0.0)
    parser.add_argument('--choice', type=str, default="ent", choices=["maxp", "ent", "marginp", "random"])
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--model', type=str, default="target", choices=["source", 'target'])
    parser.add_argument('--issave', type=bool, default=False)

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3

    if args.net == 'resnet101':
        args.batch_size = 24

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '../data/' 
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'pacs':
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_train_kfold.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_test_kfold.txt'

        args.name = names[args.s][0].upper() + names[args.t][0].upper() 
        if args.model == "source":
            args.output_dir = osp.join(args.output_tar, args.da, args.dset, names[args.s][0].upper())  
        else:
            args.output_dir = osp.join(args.output_tar, args.da, args.dset, args.name)

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        if args.dset == 'VISDA-C':
            if args.da == 'pda':
                args.class_num = 12
                args.src_classes = [i for i in range(12)]
                args.tar_classes = [i for i in range(6)]

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda' and args.cls_par > 0:
            args.savename += ('_thr' + str(args.threshold))
        if args.ssl > 0:
             args.savename += ('_ssl_' + str(args.ssl))

        if args.model == "source":
            args.savename = "srconly"

        args.log = 'ps_' + str(args.ps) + '_' + args.savename
        args.mm_dir = osp.join(args.output, args.da, args.dset, args.name)
        if not osp.exists(args.mm_dir):
            os.system('mkdir -p ' + args.mm_dir)
        if not osp.exists(args.mm_dir):
            os.mkdir(args.mm_dir)
        args.out_file = open(osp.join(args.mm_dir, "{:}_{:}.txt".format(args.log, args.choice)), "w")

        utils.print_args(args)
        txt_src, txt_tgt = split_target(args)
        train(args, txt_src, txt_tgt)

        args.out_file.close()