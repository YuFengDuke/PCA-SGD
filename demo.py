#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 14:25:54 2022

@author: yu
"""


import matplotlib.pyplot as plt
from utils_flatness import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class FC(nn.Module):
    def __init__(self,H):
        super(FC,self).__init__()
        self.net = nn.Sequential(
                Flatten(),
                nn.Linear(784,H,bias = False),
                nn.ReLU(),
                nn.Linear(H,H,bias =False),
                nn.ReLU(),
                nn.Linear(H,10,bias = False),
                )
    def forward(self,x):
        output = self.net(x)
        return output




sample_holder = [0,1,2,3,4,5,6,7,8,9]
LR = 0.1
batch_size = 50
EPOCH = 1000
test_size = 1000
sample_num  = 8000
layer_index = [1]
net_size = 50
stop_loss = 5e-5
evaluation = True

# prepare data, sample_hodler: the included digits; smaple num: number of sample per digits
train_x,train_y,test_x,test_y = sub_set_task(sample_holder,sample_num)
model = FC(net_size)
optimizer = torch.optim.SGD(model.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(torch_dataset, batch_size = batch_size, shuffle=True)

add_regulization = False
weight_holder = []
train_accuracy_holder = []
for epoch in range(EPOCH):
    model.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        # collect the trajectory
        weight_holder.append(get_layer_paras(model,layer_index[0]))
        out_put = model(b_x)
        loss = loss_func(out_put,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    model.eval()
    
    train_loss = cal_loss(model, train_x, train_y)
    train_accuracy = predict_accuracy(model,data_x = train_x,data_y = train_y)
    train_accuracy_holder.append(train_accuracy)
    
    if (epoch%1 == 0):
        print('Epoch is |',epoch,
              'train accuracy is |',train_accuracy,
              'train loss is |', train_loss)
        
    if (np.mean(train_accuracy_holder[-1:-10:-1])>0.99)&(train_loss < stop_loss):
        break
    
if evaluation == True:
    model.eval()
    num_weight = np.sum([np.prod(list(model.parameters())[l].shape) for l in layer_index])
    cut_off = 1000
    step_size = 10
    
    # components refer to eigen directions, also could use cal_fisher_information to get the components
    variance, components, pca = PCA(weight_holder,cut_off,window_size_epoch_unit = 10,epoch_size = 1200)
    
    flatness = cal_flatness(model,components,data_x = train_x,data_y = train_y,layer_index = layer_index,cut_off = cut_off,step = step_size) ######### flatness in each direction
    
    plt.figure()
    # includes the first component
    plt.plot(variance[1:cut_off:step_size], flatness)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\sigma^2$')
    plt.ylabel('flatness')
    
    
    plt.figure()
    for i in [5,20,50,100,200]:
        x,y = cal_loss_landscape_for_given_direction(model,components[i,:],data_x = train_x,data_y = train_y,search_range_n = -3,search_range_p = 3, layer_index = layer_index,search_point = 100)
        plt.plot(x,y,label = "component " + str(i))
    plt.xlabel('w')
    plt.ylabel('L')
    plt.yscale('log')
    plt.legend()
    