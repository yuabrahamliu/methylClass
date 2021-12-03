# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:51:32 2021

@author: liuy47
"""

#%% Config

import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn as nn
import torch
import torch.nn.functional as F

#from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

#Returns a bool indicating if CUDA is currently available
#This package adds support for CUDA tensor types, that implement the same function 
#as CPU tensors, but they utilize GPUs for computation
cuda = True if torch.cuda.is_available() else False


#labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
#labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')

#data_tr_list = []
#data_te_list = []
#for i in view_list:
#    data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
#    data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))


#from models import init_model_dict, init_optim

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        
        return x


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        
        if pow(self.num_cls, num_view) > 2000: 
            
            self.model = nn.Sequential(
                nn.Linear(num_cls, hvcdn_dim),
                nn.LeakyReLU(0.25),
                nn.Linear(hvcdn_dim, num_cls)
            )
            
        else:  
            self.model = nn.Sequential(
                nn.Linear(pow(num_cls, num_view), hvcdn_dim),
                nn.LeakyReLU(0.25),
                nn.Linear(hvcdn_dim, num_cls)
            )
        
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        
        
        if pow(self.num_cls, num_view) > 2000 and num_view >= 2: 
            vcdn_feat = in_list[0]*in_list[1]
            if num_view >= 3: 
                for i in range(2, num_view):
                    vcdn_feat = vcdn_feat*in_list[i]
        else:
            x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
            for i in range(2,num_view):
                x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
            vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
                
        output = self.model(vcdn_feat)

        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict



#from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.asscalar(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g

def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj

def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj



#Convert training and testing data (Each row is a sample and each column is a 
#feature) and labels
def prepare_trte_data_r(data_tr_list, 
                        data_te_list, 
                        labels_tr, 
                        labels_te):
    
    num_view = len(data_tr_list)
    
    #labels_tr = labels_tr.astype(int)
    #labels_te = labels_te.astype(int)
    
    labels_tr = list(map(int, labels_tr))
    labels_te = list(map(int, labels_te))
    
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        #concatenate((a1, a2, ...), axis = 0)
        #Join a sequence of arrays along an existing axis
        #Parameters
        #a1, a2, ... : sequence of array_like
        #  The arrays must have the same shape, except in the dimension 
        #  corresponding to `axis` (the first, by default)
        #axis: int, optional
        #  The axis along which the arrays will be joined. If axis is None, 
        #  arrays are flattened before use. Default is 0.
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels

#Calculate sample distance matrices
def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list

#Return loss of each network

#data_list = data_tr_list
#adj_list = adj_tr_list
#label = labels_tr_tensor #####
#one_hot_label = onehot_labels_tr_tensor
#sample_weight = sample_weight_tr #####
#model_dict = model_dict
#optim_dict = optim_dict
#train_VCDN = True

def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, 
                model_dict, optim_dict, train_VCDN = True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()   
    num_view = len(data_list)
    
    #Calculate the loss of each view network
    for i in range(num_view):
        
        #torch.optim.Adam
        #Implements Adam algorithm (a method for stochastic optimization)
        #Args:
        #lr: learning rate (default: 1e-3)
        #betas: coefficients used for computing running averages of gradient and 
        #       its square (default: (0.9, 0.999))
        #eps: term added to the denomiator to improve numeriacal stability 
        #     (default: 1e-8)
        #weight_decay: weight decay (L2 penalty) (default: 0)
        
        #torch.optim.Adam.zero_grad
        #Sets the gradients of all optimized `torch.Tensor`s to zero.
        optim_dict["C{:}".format(i+1)].zero_grad()
        
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        
        #backward
        #Computes the gradient of current tensor w.r.t. graph leaves
        #The graph is differentiated using the chain rule.
        #This function accumulates gradients in the leaves - you migth need to 
        #zero ``.grad`` attributes or set them to ``None`` before calling it.
        ci_loss.backward()
        
        #torch.optim.Adam.step
        #Performs a single optimization step
        optim_dict["C{:}".format(i+1)].step()
        
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
        
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
            
            
        c = model_dict["C"](ci_list)    
        
        
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict

#Return prediction matrix for the data
def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob

#main mogonet function      
def mogonet_r(data_tr_list, 
              data_te_list, 
              labels_tr, 
              labels_te, 
              num_class, 
              
              num_epoch_pretrain, 
              num_epoch, 
              lr_e_pretrain = 1e-3, 
              lr_e = 5e-4, 
              lr_c = 1e-3, 
              seednum = 1234, 
              test_inverval = 50, 
              adj_parameter = 10, 
              dim_he_list = [400, 400, 200]):
    
    #labels_tr = labels_tr.astype(int)
    #labels_te = labels_te.astype(int)
    labels_tr = list(map(int, labels_tr))
    labels_te = list(map(int, labels_te))
    
    #print(labels_te)
    #return(labels_te)


    
    num_class = int(num_class)
    num_epoch_pretrain = int(num_epoch_pretrain)
    num_epoch = int(num_epoch)
    
    seednum = int(seednum)
    test_inverval = int(test_inverval)
    adj_parameter = int(adj_parameter)
    dim_he_list = list(map(int, dim_he_list))
    
    num_view = len(data_tr_list)
    dim_hvcdn = pow(num_class,num_view)
    
    if dim_hvcdn > 2000: 
        dim_hvcdn = num_class
        
    #print(dim_hvcdn)
    #return dim_hvcdn
    

    #Load in training and testing data (Each row is a sample and each column is a 
    #feature) and labels
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data_r(data_tr_list = data_tr_list, 
                                                                              data_te_list = data_te_list, 
                                                                              labels_tr = labels_tr, 
                                                                              labels_te = labels_te)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    #Change training labels to a one-hot format
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    #Calculate sample weights to handle the class imbalance problem
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
        
    #Calculate sample distance matrices
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, \
                                                trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    #Output whole model topological structure
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain GCNs...")
    #Output parameters of the whole model for pretraining on training set
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    
    #Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seednum)
    
    #Pretraining on training set
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, 
                    model_dict, optim_dict, train_VCDN=False)
        
    print("\nTraining...")
    #Output parameters of the whole model for training on training set
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    
    #Seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seednum)

    #Training on training set
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, 
                    model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            
            print()
    
    return te_prob
            
#%% Import R data
"""
import os
import rpy2.robjects as robjects

import numpy as np
#import pandas as pd

wkdir = 'C:/Users/liuy47/Desktop/Transfer/nihres/methypreprocessing/files/ml4calibrated450k-master/'

os.chdir(wkdir)

data_tr_list_file = 'data_tr_list_mogonet.RData'
data_te_list_file = 'data_te_list_mogonet.RData'
labels_tr_file = 'labels_tr_mogonet.RData'
labels_te_file = 'labels_te_mogonet.RData'
#labels_te_file = 'labels_te_mogonet_tst.RData'


#Interface between R and Python

robjects.r['load'](data_tr_list_file)
data_tr_list = robjects.r['data_tr_list']

data_tr_list = list(data_tr_list)
data_tr_list = [np.asarray(x) for x in data_tr_list]


robjects.r['load'](data_te_list_file)
data_te_list = robjects.r['data_te_list']

data_te_list = list(data_te_list)
data_te_list = [np.asarray(x) for x in data_te_list]


robjects.r['load'](labels_tr_file)
labels_tr = robjects.r['labels_tr']

#labels_tr = np.asarray(labels_tr)
labels_tr = list(labels_tr)


robjects.r['load'](labels_te_file)
labels_te = robjects.r['labels_te']

#labels_te = np.asarray(labels_te)
labels_te = list(labels_te)


data_tr_list = data_tr_list
data_te_list = data_te_list
labels_tr = labels_tr
labels_te = labels_te
num_class = 91

num_epoch_pretrain = 100
num_epoch = 250
lr_e_pretrain = 0.001
lr_e = 5e-4
lr_c = 0.001
seednum = 1234
test_inverval = 50
adj_parameter = 10
dim_he_list = [400, 400, 200]

"""



