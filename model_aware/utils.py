import numpy as np
import sys
from scipy import special
import torch


# 获取模型之间的相似性

def get_each_difference_matrix(nets, selected, layername = "fc3.weight"):
    selected_number = len(selected)
    difference_martrix = torch.eye(selected_number)
    for i in range(selected_number):
        for j in range(i+1, selected_number):
            weight_i = nets[selected[i]].state_dict()["fc3.weight"]
            weight_j = nets[selected[j]].state_dict()["fc3.weight"]
            sim_ij = max(0,torch.cosine_similarity(weight_i, weight_j,dim=-1).mean(),)
            difference_martrix[i][j] = sim_ij
            difference_martrix[j][i] = sim_ij

    return difference_martrix

# 获取本地模型和全局模型之间的相似性

def get_central_difference_vector(global_net, nets, selected, layername="fc3.weight"):
    global_fc3_weight = global_net["fc3.weight"]
    selected_number = len(selected)
    difference_vector = [0 for i in range(selected_number)]
    for i in range(selected_number):
        net_para = nets[selected[i]].state_dict()
        local_fc3_weight = net_para["fc3.weight"]
        sim_local_global = max(0, torch.cosine_similarity(global_fc3_weight,local_fc3_weight,dim=-1).mean())
        difference_vector[i] = 1 - sim_local_global.item()
    all = sum(difference_vector)
    for i in range(selected_number):
        difference_vector[i] = difference_vector[i]/all
    difference_vector = np.asarray(difference_vector)
    return difference_vector




