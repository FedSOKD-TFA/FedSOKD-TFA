import math

import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from collections import OrderedDict, defaultdict
from pathlib import Path
import argparse
import logging
import os
import copy
from math import *
import random


import datetime
from torch.utils.tensorboard import SummaryWriter
from distribution_aware.utils import get_distribution_difference
from model_aware.utils import get_central_difference_vector


from models.cnn import  CNNTarget

from utils import *
from methods.method import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir',
                        help='the data partitioning strategy:homo/noniid-labeldir/iid-label100/noniid-labeldir100/noniid-labeluni')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='FedSOKD',help='communication strategy: FedSOKD/ FedSOKD-TFA')
    parser.add_argument('--comm_round', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.01,
                        help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=5e-4, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='None', help='Noise type: None/increasng/space')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.5, help='Sample ratio for each communication round')
    parser.add_argument('--train_acc_pre', action='store_true', default=True)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--test_round', type=int, default=2)
    parser.add_argument("--save_model", action='store_true', default=True)
    parser.add_argument("--comment", default="_")
    parser.add_argument("--definite_selection", action='store_true', default=False)
    parser.add_argument("--show_all_accuracy", action='store_true', default=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument('--log_flag', default=True)

    """
    Weight Shrinking
    """
    parser.add_argument('--weight_shrinking', type=float, default=0.95, help="Controlling the Weight Shrinking")

    """
    Used for TFA
    """
    parser.add_argument('--TFA_used', type=bool, default=True, help='Wether use the TFA method')
    parser.add_argument('--measure_difference', type=str, default='kl',
                        help='How to measure difference. e.g. only_iid, cosine')
    parser.add_argument('--distribution_aware', type=str, default='yes',
                        help='Types of distribution aware e.g. division')
    parser.add_argument('--difference_operation', type=str, default='linear',
                        help='Conduct operation on difference. e.g. square or cube')
    parser.add_argument('--TFA_a', type=float, default=0.6, help='Under sub mode, n_k-TFA_a*s_k+TFA_b*miu+TFA_c')
    parser.add_argument('--TFA_b', type=float, default=0.3)
    parser.add_argument('--TFA_c', type=float, default=0.3)

    """
    Used for SOKD
    """
    parser.add_argument("--T", type=float, default=7)
    parser.add_argument("--loss_partial",type=float,default=0.95)

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):
        if args.model == "cnn":
            if args.dataset == "cifar10":
                net = CNNTarget(n_kernels=16)
            elif args.dataset == "cifar100":
                net = CNNTarget(n_kernels=16, out_dim=100)
        else:
            raise NotImplementedError("not supported yet")

        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def init_personalized_parameters(args, client_number=None):
    personalized_pred_list = []
    if args.dataset == "cifar10":
        class_num = 10
    elif args.dataset == "cifar100":
        class_num = 100

    if args.alg == 'FedSOKD':
        if args.model == 'cnn':
            dim = 84
            for nndx in range(args.n_parties):
                para_dict = OrderedDict()
                para_dict["fc3.weight"] = None
                para_dict["fc3.bias"] = None
                personalized_pred_list.append(para_dict)

    return personalized_pred_list

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map_train, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map_train

if __name__ == '__main__':
    args = get_args()
    logging.info("Dataset: %s" % args.dataset)
    logging.info("Backbone: %s" % args.model)
    logging.info("Method: %s" % args.alg)
    logging.info("Partition: %s" % args.partition)
    logging.info("Beta: %f" % args.beta)
    logging.info("Sample rate: %f" % args.sample)
    logging.info("Using Optimizer: %s" % args.optimizer)
    logging.info("Print Accuracy on training set: %s" % args.train_acc_pre)
    logging.info("Save model: %s" % args.save_model)
    logging.info("Total running round: %s" % args.comm_round)
    logging.info("Test round fequency: %d" % args.eval_step)
    logging.info("Show every client's accuracy: %s" %args.show_all_accuracy)

    save_path = args.alg+"-"+"-"+args.model+"-"+str(args.n_parties)+"-"+args.dataset+"-"+args.partition+args.comment
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path = args.alg + "-" + args.model + "-" + args.dataset + "-" + args.partition + str(
            args.version) + '-experiment_arguments-%s.json ' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = args.model + " " + str(args.version) + '-experiment_log-%s ' % (
            datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = set_logger(args)
    logger.setLevel(logging.INFO)
    logger.info(device)
    logger.info("Dataset: %s" % args.dataset)
    logger.info("Backbone: %s" % args.model)
    logger.info("Method: %s" % args.alg)
    logger.info("Partition: %s" % args.partition)
    logger.info("Beta: %f" % args.beta)
    logger.info("Sample rate: %f" % args.sample)
    logger.info("Using Optimizer: %s" % args.optimizer)
    logger.info("Print Accuracy on training set: %s" % args.train_acc_pre)
    logger.info("Save model: %s" % args.save_model)
    logger.info("Total running round: %s" % args.comm_round)
    logger.info("Test round fequency: %d" % args.eval_step)
    logger.info("Noise Type: %s" % args.noise_type)
    logger.info("Show every client's accuracy: %s" % args.show_all_accuracy)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    eval_step = args.eval_step
    acc_all = []

    logger.info("Partitioning data")

    logging.info("Test beginning round: %d" % args.test_round)
    logging.info("Client Number: %d" % args.n_parties)
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta, logdir=args.logdir)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,
                                                                                      args.batch_size, 32)
    logger.info("len train_dl_global: %d" % len(train_ds_global))
    data_size = len(test_ds_global)

    results_dict = defaultdict(list)
    eval_step = args.eval_step
    best_step = 0
    best_accuracy = -1
    test_round = args.test_round

    if args.alg =='FedSOKD':
        if args.model not in ['cnn']:
            raise NotImplementedError("FedSOKD-TFA uses cnn as backbone")
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        best_nets = copy.deepcopy(nets)  # Initialise to nets
        rounds_acc = np.zeros(args.n_parties)  # Initialise an array of accuracies for each round of participation in training the network
        best_acc = np.zeros(args.n_parties)  # Initialise the array of best accuracies for each round

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        logger.info("Initializing Personalized Classification head")
        personalized_pred_list = init_personalized_parameters(args)

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        if args.TFA_used == True:
            # Here traindata_cls_counts is in a dictionary format, while the first parameter of get_distribution_difference
            # below requires the format to be a two-dimensional array
            # Here  need to convert traindata_cls_counts into a two-dimensional array

            row = args.n_parties
            cols = n_classes
            traindata_cls_counts_list = [[traindata_cls_counts[i].get(j, 0) for j in range(cols)] for i in range(row)]
            traindata_cls_counts_array = np.array(traindata_cls_counts_list)
            paritial = np.sum(traindata_cls_counts_array, axis=1) / np.sum(traindata_cls_counts_array)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    node_weights = personalized_pred_list[idx]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)

            _, rounds_acc = local_train_net_with_distillation(nets, best_nets, selected, args, net_dataidx_map_train,
                                                              net_dataidx_map_test,
                                                              logger,
                                                              device=device)
            # update global model

            total_data_points = sum([len(net_dataidx_map_train[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map_train[r]) / total_data_points for r in selected]

            if args.TFA_used == True:
                print(f'Data Partial : {fed_avg_freqs}')
                # calculate the discrepancy
                distribution_difference = get_distribution_difference(traindata_cls_counts_array,
                                                                      partipation_clients=selected,
                                                                      metric=args.measure_difference)
                if np.sum(distribution_difference) == 0:
                    distribution_difference = np.array([0 for _ in range(len(distribution_difference))])
                else:
                    distribution_difference = distribution_difference / np.sum(
                        distribution_difference)  # normalize. (some metrics make the difference value larger than 1.0)
                if args.difference_operation == 'linear':
                    pass
                elif args.difference_operation == 'square':
                    distribution_difference = np.power(distribution_difference, 2)
                elif args.difference_operation == 'cube':
                    distribution_difference = np.power(distribution_difference, 3)
                else:
                    raise NotImplementedError
                if round == 0 or args.sample < 1.0:
                    print(f'Distribution_difference : {distribution_difference}')

                model_difference_vector = get_central_difference_vector(global_para, nets, selected)
                print(f'model_difference_vector: {model_difference_vector}')

                # adjusting the aggregation weight
                if args.distribution_aware != 'not':
                    a = args.TFA_a
                    b = args.TFA_b
                    c = args.TFA_c
                    tmp = fed_avg_freqs - a * distribution_difference - b * model_difference_vector + c

                    if np.sum(tmp > 0) > 0:  # ensure not all elements are smaller than 0
                        fed_avg_freqs = np.copy(tmp)
                        fed_avg_freqs[fed_avg_freqs < 0.0] = 0.0
                    total_normalizer = sum([fed_avg_freqs[r] for r in range(len(selected))])
                    fed_avg_freqs = [fed_avg_freqs[r] / total_normalizer for r in range(len(selected))]
                    if round == 0 or args.sample < 1.0:
                        print(f'TFA Weights : {fed_avg_freqs}')

            if round == 0:
                for iidx in range(args.n_parties):
                    final_state = copy.deepcopy(nets[iidx].state_dict())
                    if args.model == 'cnn':
                        personalized_pred_list[iidx]["fc3.weight"] = copy.deepcopy(final_state["fc3.weight"])
                        personalized_pred_list[iidx]["fc3.bias"] = copy.deepcopy(final_state["fc3.bias"])

            for idx in range(len(selected)):
                if round != 0:
                    final_state = copy.deepcopy(nets[selected[idx]].state_dict())
                    if args.model == 'cnn':
                        personalized_pred_list[selected[idx]]["fc3.weight"] = copy.deepcopy(
                            final_state["fc3.weight"])
                        personalized_pred_list[selected[idx]]["fc3.bias"] = copy.deepcopy(
                            final_state["fc3.bias"])

                net_para = nets[selected[idx]].state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx] * args.weight_shrinking
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx] * args.weight_shrinking

            global_model.load_state_dict(global_para)

            if (round + 1) >= test_round and (round + 1) % eval_step == 0:
                train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc, tmp_acc = compute_accuracy_personally(
                    personalized_pred_list, global_model, args, net_dataidx_map_train, net_dataidx_map_test,
                    device=device)
                for idx in range(args.n_parties):
                    if (tmp_acc[idx] > best_acc[idx]):
                        best_acc[idx] = tmp_acc[idx]
                        node_weights = personalized_pred_list[idx]
                        best_nets[idx].load_state_dict(global_para)
                        best_nets[idx].load_state_dict(node_weights, strict=False)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' % test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc * 100)
                results_dict['test_all_acc'].append(test_all_acc)

        _, _, _, _, _, _, test_acc, _, best_test_acc = compute_accracy_personally_best(
            best_nets, args, net_dataidx_map_train, net_dataidx_map_test,
            device=device)
        logger.info("best_test_acc:%s" % best_test_acc)
        logger.info("best_acc:%s" % best_acc)
        logger.info("best_acc:%f" % test_acc)

        save_path = Path("results_table/" + save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.special_normalize) + "-" + str(args.n_parties) + "-" + str(
            args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch': args.comm_round + 1, 'state': global_model.state_dict()}, outfile_gmodel)
            for ele in range(len(personalized_pred_list)):
                p_head = os.path.join(save_path, 'phead_1500_' + str(ele) + ".tar")
                torch.save({'epoch': args.comm_round + 1, 'state': personalized_pred_list[ele]}, p_head)

        json_file_opt = "results_" + accessories + ".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)























