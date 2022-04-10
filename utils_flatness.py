#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:21:25 2022

@author: yu
"""


import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy import io
import scipy
import torch.nn as nn
import copy
import torchvision
import torch.nn.functional as F
from scipy import linalg


def shuffle_data_set(data_x,data_y):
    np.random.seed(1)
    index = np.arange(0,len(data_y))
    index = np.random.permutation(index)
    data_x = data_x[index,:]
    data_y = data_y[index]
    return data_x,data_y

def sub_set_task(label_list,sample_number):
    
    download_mnist = True
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',    
        train=True,  
        transform = torchvision.transforms.ToTensor(),                                                      
        download = download_mnist,
    )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    
    
    Train_x = train_data.data.type(torch.FloatTensor).view(-1,28*28)/255.
    Train_y = train_data.targets
    Test_x = test_data.test_data.type(torch.FloatTensor).view(-1,28*28)/255.   # shape from (2000, 28, 28) to (2000, 784), value in range(0,1)
    Test_y = test_data.targets
    
    Train_x,Train_y = shuffle_data_set(Train_x,Train_y)
    train_index = []

    count = 0
    for i in range(len(Train_x)):
        if Train_y[i] in label_list:
            train_index.append(i)
            count = count+1
        if count >= sample_number*len(label_list):
            break
    
    test_index = []
    for i in range(len(Test_x)):
        if Test_y[i] in label_list:
            test_index.append(i)

    
    
    Train_y = np.array(Train_y)
    train_x = Train_x[np.array(train_index),:]
    train_y = Train_y[np.array(train_index)]
    
    train_x = torch.tensor(train_x).type(torch.FloatTensor)
    train_y = torch.tensor(train_y).type(torch.LongTensor)

    Test_y = np.array(Test_y)
    test_x = Test_x[np.array(test_index),:]
    test_y = Test_y[np.array(test_index)]
    
    test_x = torch.tensor(test_x).type(torch.FloatTensor)
    test_y = torch.tensor(test_y).type(torch.LongTensor)
    
    train_x = train_x.view(-1,1,28,28)
    test_x = test_x.view(-1,1,28,28)
    
    return train_x,train_y,test_x,test_y

def predict_accuracy(model,data_x,data_y):
    
    pred = torch.max(model(data_x),1)[1].data.numpy()
    accuracy = np.mean(pred == data_y.data.numpy())
    
    return accuracy

def cal_loss(model,data_x,data_y):

    # Send things to the right place
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    
    loss_func = nn.CrossEntropyLoss()
    out_put = model(data_x)
    loss = loss_func(out_put,data_y)
    
    return loss.data.cpu().numpy()

def transform_matrix_to_array(para_list):
    para_holder = []
    for para in para_list:
        para_holder.append(para.data.clone().cpu().numpy().reshape(1,-1))
    para_array = np.hstack(para_holder)
    return para_array

def transform_array_to_matrix(model,layer_index,para_array):
    para_list = []
    start_point = 0
    for i in layer_index:
        weight = list(model.parameters())[i]
        num_weight = np.prod(weight.shape)
        para_matrix = para_array[0][start_point:num_weight+start_point].reshape(weight.shape)
        para_list.append(torch.tensor(para_matrix))
        start_point += num_weight
    return para_list

def get_layer_paras(fc,layer):
    paras = list(fc.named_parameters())
    name,Para = paras[layer]
    para = copy.deepcopy(Para)
    para = torch.squeeze(para.reshape(1,-1))
    para = para.data.numpy()
    return para

def get_net_paras(model):
    # model_copy = copy.deepcopy(model)
    weight_list = list(model.parameters())
    grad_list = []
    for i in range(len(weight_list)):
        grad_list.append(weight_list[i].grad.detach())
    return weight_list, grad_list

############## calculate the fisher information matrix
def cal_fisher_information(model, data_x, data_y, layer_index):

    # Send things to the right place
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # Initialize stuff
    torch_dataset = Data.TensorDataset(data_x, data_y)
    train_loader = Data.DataLoader(torch_dataset, batch_size=1, shuffle=False, pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Main loop
    fisher_info_matrix = 0
    for b_x, b_y in train_loader:

        b_x, b_y = b_x.to(device), b_y.to(device)

        # Get ouput and Loss
        out = model(b_x)
        loss = nn.CrossEntropyLoss()(out, b_y)

        # Compute Gradient
        optimizer.zero_grad()
        loss.backward()
        
        # Store gradients
        layer_weight_grad = []
        for i in layer_index:
            layer_weight_grad.append(list(model.parameters())[i].grad)
        
        # Change size
        sample_grad = transform_matrix_to_array(layer_weight_grad)

        # Compute contribution to fisher info matrix
        fisher_info_matrix += np.dot(sample_grad.T, sample_grad)

    fisher_info_matrix = fisher_info_matrix / len(train_loader)
    
    # Get eigenvalues
    v,w = linalg.eig(fisher_info_matrix, )

    model = model.to(torch.device("cpu"))

    return fisher_info_matrix, np.real(v), np.real(w).T


########### functions to calculate flatness
def replace_given_weight(weight,model,layer):
    
    model_new = copy.deepcopy(model)
    weight_new = weight
    paras = model_new.state_dict()
    paras_key = list(paras.keys())
    paras[paras_key[layer]] = weight_new
    model_new.load_state_dict(paras)
    return model_new

def amend_weight(model, weight_old_list, component, amend_amplitude, layer_index):
    new_weight_list = []
    component_list = transform_array_to_matrix(model, layer_index, component.reshape(1,-1))
    for weight_old,component in zip(weight_old_list, component_list):
        weight_change = amend_amplitude * component
        new_weight = weight_old + weight_change
        new_weight_list.append(new_weight)
    return new_weight_list

def move_model_in_weight_space(model, direction, amplitude, layer_index):
    model_new = copy.deepcopy(model)
    weight_list_start = [list(model.parameters())[l] for l in layer_index]
    new_weight_list = amend_weight(model, weight_list_start, direction, amend_amplitude = amplitude, layer_index = layer_index)
    for l in range(len(layer_index)):
        new_weight = new_weight_list[l]
        model_new = replace_given_weight(new_weight, model = model_new, layer = layer_index[l])
    return model_new
        
def loss_change_by_changing_weight(model, weight_list_start, direction, amplitude, data_x, data_y, layer_index):
    model_new = copy.deepcopy(model)
    # weight_list_start = [list(model.parameters())[l] for l in layer_index]
    new_weight_list = amend_weight(model, weight_list_start, direction, amend_amplitude = amplitude, layer_index = layer_index)  ##### move the weight in a given direction
    for l in range(len(layer_index)):
        new_weight = new_weight_list[l]
        model_new = replace_given_weight(new_weight, model = model_new, layer = layer_index[l])   ############ update new weight for each layer
    amend_loss = cal_loss(model_new, data_x, data_y)                                            ################ calculate loss for the updated model

    return amend_loss

################# binary search the range of weight when loss is below a threshold
def binary_search_loss_bound(model, direction, data_x, data_y, layer_index):
    
    s_theta = 0
    l_theta = 100
    accuracy = 0.01
    weight_list_start = [list(model.parameters())[l] for l in layer_index]
    l0 = np.log(
        loss_change_by_changing_weight( model = model, 
                                        weight_list_start=weight_list_start, 
                                        direction = direction, 
                                        amplitude = 0, 
                                        data_x = data_x, 
                                        data_y = data_y, 
                                        layer_index = layer_index)
        )
    threshold = 1

    ########## forward search
    while(1):
        loss = loss_change_by_changing_weight(  model = model,
                                                weight_list_start=weight_list_start,
                                                direction = direction,
                                                amplitude = l_theta,
                                                data_x = data_x,
                                                data_y = data_y,
                                                layer_index = layer_index)
        if np.log(loss) > l0 + threshold:
            break
        l_theta += 100
        if abs(l_theta)>1000:
            return abs(l_theta)
    while(1):
        m_theta = (s_theta+l_theta)/2
        loss = loss_change_by_changing_weight(  model = model,
                                                weight_list_start=weight_list_start,
                                                direction = direction,
                                                amplitude = m_theta,
                                                data_x = data_x,
                                                data_y = data_y,
                                                layer_index = layer_index)
        if np.log(loss) > l0 + threshold :
            l_theta = copy.deepcopy(m_theta)
        else:
            s_theta = copy.deepcopy(m_theta)
        if abs(l_theta - s_theta) < accuracy:
            break
    upper_theta = (l_theta + s_theta)/2
    
    l_theta = 0
    s_theta = -100

    ##################backward search
    while(1):
        loss = loss_change_by_changing_weight(  model = model,
                                                weight_list_start=weight_list_start,
                                                direction = direction,
                                                amplitude = s_theta,
                                                data_x = data_x,
                                                data_y = data_y,
                                                layer_index = layer_index)
        if np.log(loss) > l0 + threshold:
            break
        s_theta += -100
        if abs(s_theta)>1000:
            return abs(s_theta)
    while(1):
        m_theta = (s_theta+l_theta)/2
        loss = loss_change_by_changing_weight(  model = model,
                                                weight_list_start=weight_list_start,
                                                direction = direction,
                                                amplitude = m_theta,
                                                data_x = data_x,
                                                data_y = data_y,
                                                layer_index = layer_index)
        if np.log(loss) > l0 + threshold :
            s_theta = copy.deepcopy(m_theta)
        else:
            l_theta = copy.deepcopy(m_theta)
        if abs(l_theta - s_theta) < accuracy:
            break
    lower_theta = (l_theta + s_theta)/2
    
    return abs(upper_theta) + abs(lower_theta)

################## calculate flatness for all directions
def cal_flatness(model, components, data_x, data_y, layer_index, cut_off, step):

    # Send things to the right place
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_x, data_y = data_x.to(device), data_y.to(device)    

    flatness = []
    for i in range(0, cut_off, step):
        print('calculating flatness in {}th direction'.format(i))
        flatness.append(binary_search_loss_bound(model, components[i,:], data_x, data_y, layer_index))
    return flatness

################ visualize landscape for a given direction
def cal_loss_landscape_for_given_direction(model, direction, data_x, data_y, search_range_n, search_range_p, layer_index, search_point):
    weight_list_start = [list(model.parameters())[l] for l in layer_index]
    change_loss = []
    for i in np.linspace(search_range_n, search_range_p, search_point):
        change_loss.append( loss_change_by_changing_weight( model = model, 
                                                            weight_list_start=weight_list_start,
                                                            direction = direction,
                                                            amplitude = i,
                                                            data_x = data_x,
                                                            data_y = data_y,
                                                            layer_index = layer_index)
                            )   
    return np.linspace(search_range_n, search_range_p, search_point), change_loss

#### perform PCA on trajectories
def PCA(weight_holder,component,window_size_epoch_unit = 1,epoch_size = 1200):
    import sklearn.decomposition
    print('Calculating Principle Components...')
    # determine the window size
    window = int(epoch_size*window_size_epoch_unit)
    pca_dimension = sklearn.decomposition.PCA(n_components = component)
    weight = weight_holder[len(weight_holder)-window:len(weight_holder)-1]
    pca_dimension.fit(np.array(weight))
    variance = pca_dimension.explained_variance_
    component = pca_dimension.components_
    return variance,component,pca_dimension


###################### Examples
################# first load the trained model, train_x, train_y

############ calculate the flatness on the basis of eigenvectors of fisher matrix
# layer_index = [2] ################ gives the layers that need to analyze, it could be multiple layers 
# num_weight = np.sum([np.prod(list(model.parameters())[l].shape) for l in layer_index]) ########## total number of directions
# cut_off = num_weight
# matrix,variance,components = cal_fisher_information(model,data_x = train_x,data_y = train_y,layer_index = layer_index) ########  eigenvector and eigenvalue of fisher matrix
# flatness = cal_flatness(model,components,data_x = train_x,data_y = train_y,layer_index = layer_index,cut_off = cut_off,step = 1) ######### flatness in each direction


# ################# visualize the landscape in each firection
# plt.figure()
# for i in [0,5,10,20,100]:
#     x,y = cal_loss_landscape_for_given_direction(model,components[i,:],data_x = train_x,data_y = train_y,search_range_n = -10,search_range_p = 10,layer_index = layer_index,search_point = 100)
#     plt.plot(x,y,label = "component " + str(i))
# plt.xlabel('w')
# plt.ylabel('L')
# plt.yscale('log')
# plt.legend()

