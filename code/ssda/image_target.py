import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import rotation
from loss import CrossEntropyLabelSmooth

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size // 2

    dsets["target_l"] = ImageList_idx(open(args.tl_dset_path).readlines(), transform=image_train())
    dsets["target_u"] = ImageList_idx(open(args.tu_dset_path).readlines(), transform=image_train())
    dsets["test"] = ImageList_idx(open(args.tu_dset_path).readlines(), transform=image_test())

    dset_loaders["target_l"] = DataLoader(dsets["target_l"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["target_u"] = DataLoader(dsets["target_u"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
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
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    return accuracy*100, mean_ent

def cal_acc_rot(loader, netF, netB, netR):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            r_labels = np.random.randint(0, 4, len(inputs))
            r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
            r_labels = torch.from_numpy(r_labels)
            r_inputs = r_inputs.cuda()
            
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))

            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
            if start_test:
                all_output = r_outputs.float().cpu()
                all_label = r_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, r_labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    return accuracy*100

def train_target_rot(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netR = network.feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    netF.eval()
    for k, v in netF.named_parameters():
        v.requires_grad = False
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    netB.eval()
    for k, v in netB.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*1}]
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target_u"])
    interval_iter = max_iter // 10
    iter_num = 0

    rot_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target_u"])
            inputs_test, _, tar_idx = iter_test.next()

        try:
            inputs_target_l, _, tar_l_idx = iter_target_l.next()
        except:
            iter_target_l = iter(dset_loaders["target_l"])
            inputs_target_l, _, tar_l_idx = iter_target_l.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_target = torch.cat((inputs_test, inputs_target_l), 0).cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        r_labels_target = np.random.randint(0, 4, len(inputs_target))
        r_inputs_target = rotation.rotate_batch_with_labels(inputs_target, r_labels_target)
        r_labels_target = torch.from_numpy(r_labels_target).cuda()
        r_inputs_target = r_inputs_target.cuda()
        
        f_outputs = netB(netF(inputs_target))
        f_r_outputs = netB(netF(r_inputs_target))
        r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

        rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
        rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netR.eval()
            acc_rot = cal_acc_rot(dset_loaders['target_u'], netF, netB, netR)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_rot)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netR.train()

            if rot_acc < acc_rot:
                rot_acc = acc_rot
                best_netR = netR.state_dict()

    log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return best_netR, rot_acc

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda() 

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    if not args.ssl == 0:
        netR = network.feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck).cuda()
        netR_dict, acc_rot = train_target_rot(args)
        netR.load_state_dict(netR_dict)
    
    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    if not args.ssl == 0:
        for k, v in netR.named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        netR.train()

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target_u"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target_u"])
            inputs_test, _, tar_idx = iter_test.next()

        try:
            inputs_target_l, labels_target_l, tar_l_idx = iter_target_l.next()
        except:
            iter_target_l = iter(dset_loaders["target_l"])
            inputs_target_l, labels_target_l, tar_l_idx = iter_target_l.next()

        if inputs_test.size(0) == 1 or inputs_target_l.size(0) == 1:
            continue
            
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args, dset_loaders['target_l'])
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        if args.cls_par > 0:
            pred = mem_label[tar_idx]

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        inputs_target_l = inputs_target_l.cuda()
        features_target_l = netB(netF(inputs_target_l))
        outputs_target_l = netC(features_target_l)
        classifier_loss += CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_target_l, labels_target_l.cuda())

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        classifier_loss.backward()

        if not args.ssl == 0:
            r_labels_target = np.random.randint(0, 4, len(inputs_test))
            r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()

            f_outputs = netB(netF(inputs_test))
            f_outputs = f_outputs.detach()
            f_r_outputs = netB(netF(r_inputs_target))
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = args.ssl * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)   
            rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args, labeled_dset=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    if labeled_dset: 
        start_test = True
        with torch.no_grad():
            iter_test = iter(labeled_dset)
            for _ in range(len(labeled_dset)):
                data = iter_test.next()
                inputs = data[0].cuda()
                feas = netB(netF(inputs))
                labels = data[1]
                if start_test:
                    all_fea_l = feas.float().cpu()
                    all_label_l = labels
                    start_test = False
                else:
                    all_fea_l = torch.cat((all_fea_l, feas.float().cpu()), 0)
                    all_label_l = torch.cat((all_label_l, labels), 0)

        if args.distance == 'cosine':
            all_fea_l = torch.cat((all_fea_l, torch.ones(all_fea_l.size(0), 1)), 1)
            all_fea_l = (all_fea_l.t() / torch.norm(all_fea_l, p=2, dim=1)).t()

        all_fea = torch.cat((all_fea, all_fea_l), 0)
        all_output = torch.cat((all_output, torch.eye(args.class_num)[all_label_l]), 0)

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    predict = predict[0:len(all_label)]
    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-home'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vgg16', choices=["resnet34", "vgg16", "resnet50"])
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--ssl', type=float, default=0.0) 
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--shot', type=int, default=1, choices=[1, 3])
    args = parser.parse_args()
    
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.class_num = 65 
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '../data/ssda/'
    args.s_dset_path = folder + args.dset + '/labeled_source_images_' + names[args.s]  + '.txt'
    args.tl_dset_path = folder + args.dset + '/labeled_target_images_' + names[args.t] + '_' + str(args.shot) + '.txt'
    args.tu_dset_path = folder + args.dset + '/unlabeled_target_images_' + names[args.t] + '_' + str(args.shot) + '.txt'

    args.output_dir_src = osp.join(args.output_src, 'ssda', args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, 'shot_' + str(args.shot), args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.ssl > 0:
         args.savename += ('_ssl_' + str(args.ssl))
    
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    train_target(args)

    args.out_file.close()