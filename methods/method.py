import numpy as np
import json
import torch
import torch.optim as optim
from collections import OrderedDict, defaultdict
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

from utils import *

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0
    net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "cifar10":
        num_class = 10

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            sample_per_class = torch.zeros(num_class)
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                cnt += 1
                epoch_loss_collector.append(loss.item())

        if len(epoch_loss_collector) == 0:
            assert args.model in ['cnn-b']
            epoch_loss = 0
        else:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    if args.train_acc_pre:
        train_acc = compute_accuracy(net, train_dataloader, device=device)
        return train_acc, test_acc
    else:
        return None, test_acc

def train_net_with_distillation(net_id , net, best_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    T = args.T
    cnt = 0
    net.train()
    best_net.eval()

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "cifar10":
        num_class = 10

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            sample_per_class = torch.zeros(num_class)
            for batch_idx, (x, target) in enumerate(tmp):

                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out1 = net(x)
                out2 = best_net(x)
                loss1 = criterion(out1,target)
                soft_loss = nn.KLDivLoss(reduction="batchmean")
                loss2 = soft_loss(F.softmax(out1/T,dim=1),F.softmax(out2/T,dim=1))
                loss = loss1*args.loss_partial + loss2*(1-args.loss_partial)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
        if len(epoch_loss_collector) == 0:
            assert args.model in ['cnn-b']
            epoch_loss = 0
        else:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        if args.train_acc_pre:
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            return train_acc, test_acc
        else:
            return None, test_acc

def local_train_net_with_distillation(nets, best_nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    acc = np.zeros(args.n_parties)
    avg_acc = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        best_net = best_nets[net_id]
        net.to(device)
        best_net.to(device)
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0
        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir,
                                                                         args.batch_size, 32, dataidxs_train,
                                                                         dataidxs_test, noise_level, net_id,
                                                                         args.n_parties - 1)
        elif args.noise_type == 'increasing':
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir,
                                                                         args.batch_size, 32, dataidxs_train,
                                                                         dataidxs_test, noise_level,
                                                                         apply_noise=True)
        else:
            noise_level = 0
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir,
                                                                         args.batch_size, 32, dataidxs_train,
                                                                         dataidxs_test, noise_level)

        trainacc, testacc = train_net_with_distillation(net_id, net, best_net, train_dl_local, test_dl_local, n_epoch,
                                                        args.lr, args.optimizer, args, device=device)

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc[net_id] = testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, acc











