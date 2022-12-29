import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class DiskDataset(Dataset):
    def __init__(self, datalist_path, data_path, model, data_len, pre_len, pos_dim,train = True):
#         self.datalist_path = datalist_path
#         self.datalist = self.dataload(datalist_path)
#         self.data_path = data_path
#         self.model = model
#         self.pos_map = self.position_map(pos_dim, int(data_len/2))
#         self.data_path = './dataprocess/AMENDER/new_data_'+str(data_len)+'_'+str(pre_len)+'/'
#         self.data_path = './dataprocess/AMENDER/ST-2/'+str(data_len)+'_'+str(pre_len)+'/'
        self.data_path = './dataprocess/AMENDER/new_data_'+str(data_len)+'/'
        if train:
            self.data = np.load(self.data_path+'train_data.npy')
            self.label = np.load(self.data_path+'train_label.npy')
        else:
            self.data = np.load(self.data_path+'test_data.npy')
            self.label = np.load(self.data_path + 'test_label.npy')
    def __getitem__(self, index):
        data = self.data[index]
        half_row = int(data.shape[0]/2)
        part1 = data[0:half_row]
        part2 = data[half_row:]
        return torch.from_numpy(np.array(part1)), torch.from_numpy(np.array(part2)), torch.from_numpy(np.array(self.label[index]))
        
        
        
        
        
#         file_name, label = self.datalist[index].split(" ")
# #         path = path.split("./")[1]
#         file_type = '.npy'
#         #分割成前n天和后n天
#         data = np.load(self.data_path + file_name + file_type)
#         col = data.shape[1] #列数，表示SMART的数量
#         row = data.shape[0] #行数，表示采集的长度
#         half_row = int(row/2)
        
#         part1 = data[0:half_row]
#         part2 = data[half_row:]
#         if self.model is 'cnn':
#             part1 = part1.T
#             part2 = part2.T
#             part1 = part1[np.newaxis,:]
#             part2 = part2[np.newaxis,:]
#         elif self.model is 'cnn_position':
#             part1 = np.concatenate((part1, self.pos_map), axis = 1)
#             part2 = np.concatenate((part2, self.pos_map), axis = 1)
            
#             part1 = part1.T
#             part2 = part2.T
#             part1 = part1[np.newaxis,:]
#             part2 = part2[np.newaxis,:]
#         elif self.model is 'cnn_position_attention':
            
#             part1 = np.concatenate((part1, self.pos_map), axis = 1)
#             part2 = np.concatenate((part2, self.pos_map), axis = 1)
            
#         elif self.model is 'lstm':
#             pass
#         part1_tensor = torch.from_numpy(part1)
#         part2_tensor = torch.from_numpy(part2)
#         label = label.split("\n")[0]
#         label = np.array(int(label))
#         return part1_tensor, part2_tensor, torch.from_numpy(label)
         
    def __len__(self):
        
        return self.data.shape[0]
    
#     def dataload(self, datalist_path):
#         #从txt文件中，拿到每个路径和标签
#         f = open(datalist_path, 'r')
#         datalist = f.readlines()
#         return datalist
    
#     def position_map(self, d_model, positions):
#         position_mp = np.array([
#             [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]
#             for position in range(positions)])
#         position_mp[:, 0::2] = np.sin(position_mp[:, 0::2])
#         position_mp[:, 1::2] = np.cos(position_mp[:, 1::2])
#         return position_mp
    
