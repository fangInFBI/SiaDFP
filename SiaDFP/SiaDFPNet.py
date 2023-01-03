import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ruptures as rpt


class SiameseNet_pos_att_cnn_adjust(nn.Module):
    def __init__(self, numOfSmart, position_dim, data_len, kernel_s, model):
        super(SiameseNet_pos_att_cnn_adjust, self).__init__()
        self.channel = 7

        self.data_len = data_len
        self.model = model
        self.fc_input = self.channel * 2 * int(((data_len/2 - kernel_s + 1)/kernel_s - int(kernel_s/2) + 1)/int(kernel_s/2))
        
        self.cnnlstm_cnnoutput = 16
        self.cnnlstm_ = self.cnnlstm_cnnoutput * int((numOfSmart - 4 + 1)/2)
        
        if 'cp' in self.model and self.data_len > 10:
            self.input_channel = 2
        else:
            self.input_channel = 1
        
        #2D-Attention
        if 'att' in self.model:
            self.fearture_dim_x = numOfSmart
            self.w_x_q = nn.Parameter(torch.Tensor(
                numOfSmart, numOfSmart))
            self.w_x_k = nn.Parameter(torch.Tensor(int(data_len/2), 1))

            self.w_y_q = nn.Parameter(torch.Tensor(int(data_len/2), int(data_len/2)))
            self.w_y_k = nn.Parameter(torch.Tensor(1,numOfSmart))

            nn.init.xavier_uniform_(self.w_x_q)
            nn.init.xavier_uniform_(self.w_x_k)

            nn.init.xavier_uniform_(self.w_y_q)
            nn.init.xavier_uniform_(self.w_y_k)
        
        #Position
        self.pos_map = nn.Embedding(int(data_len/2), numOfSmart, padding_idx = 0)
        nn.init.xavier_uniform_(self.pos_map.weight)


        self.cnnlstm_seq = nn.Sequential(
            nn.Conv2d(self.input_channel, self.cnnlstm_cnnoutput, (1,4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.cnnlstm_cnnoutput),
            nn.Dropout2d(p=.3)
        )
        self.cnnlstm_pool = nn.MaxPool2d((1,2))
        self.cnnlstm_flatten1 = nn.Flatten(2)
        self.cnnlstm_flatten2 = nn.Flatten()
        
        if 'cnnlstm' in self.model:

            self.cnnlstm_lstm = nn.LSTM(self.cnnlstm_, 16,2)
            self.cnnlstm_linear = nn.Sequential(
                 nn.Linear(16*int(data_len/2),8*int(data_len/2)),
                 nn.Dropout2d(p=.4),
                 nn.LeakyReLU(),
                 nn.Linear(8*int(data_len/2), 8*int(data_len/2)),
                 nn.Dropout2d(p=.4),
                 nn.LeakyReLU(),
                 nn.Linear(8*int(data_len/2), 20)
                
            )
        
        
    def forward_once(self, x):
        x = x.type(torch.FloatTensor)
        x = x.cuda()
        output = x
        if 'att' in self.model:
            w_x_k, w_x_q,w_y_k, w_y_q = self.w_x_k, self.w_x_q, self.w_y_k, self.w_y_q
            x_k = torch.matmul(output.permute(0,2,1), w_x_k)
            x_q = torch.matmul(output, w_x_q)

            score_x = F.softmax(torch.matmul(x_q, x_k),dim = 1)
            score_x = score_x*10
            y_k = torch.matmul(w_y_k,output.permute(0,2,1))
            y_q = torch.matmul(w_y_q,output)
            y_ = torch.matmul(y_k, y_q)
            y_ = y_.squeeze(1)
            score_y = F.softmax(y_,dim = 1)
            score_y = score_y.unsqueeze(1)
            score_y = score_y*10
            score_xy = torch.matmul(score_x, score_y)
            output = torch.mul(output, score_xy)
        if 'pos' in self.model:
            idx = torch.arange(int(self.data_len/2), device = x.device)
            pos_map = self.pos_map(idx)
            output = pos_map + output

        if 'cp' in self.model and self.data_len > 10:
            CP_map = self.CPD(output)
            CP_map = CP_map.cuda()
            output = torch.stack((CP_map, output),1)
        else: 
            output = output.unsqueeze(1)


        if 'cnnlstm' in self.model:
            output = self.cnnlstm_seq(output)
            output = self.cnnlstm_pool(output)
            output = output.permute(0,2,1,3)
            output = self.cnnlstm_flatten1(output)
            output,(h,c) =self.cnnlstm_lstm(output)
            output = self.cnnlstm_flatten2(output)
            output = self.cnnlstm_linear(output)
            return output
            
        return output
      
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def position_map(self, d_model, positions):
        position_mp = np.array([
            [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]
            for position in range(positions)])
        position_mp[:, 0::2] = np.sin(position_mp[:, 0::2])
        position_mp[:, 1::2] = np.cos(position_mp[:, 1::2])
        return position_mp

    def CPD(self, data):
        datas = data.permute(0,2,1).cpu().detach().numpy()
        mark_map = np.zeros((data.shape[0],data.shape[2],data.shape[1]))
        for i, data_ in enumerate(datas):
            for j , data in enumerate(data_):
                if self.data_len <=10:
                    algo = rpt.Window(width = int(self.data_len/2) - 1, min_size = 1, jump = 1).fit(data)
                    index_changes = algo.predict(n_bkps = 1)
                elif self.data_len <= 20:
                    algo = rpt.Window(width = int(self.data_len/2) - 3, min_size = 3, jump = 1).fit(data)
                    index_changes = algo.predict(n_bkps = 1)
                else:
                    algo = rpt.Window(width = 14, min_size = 3,jump = 1).fit(data)
                    index_changes = algo.predict(n_bkps = 3)

                index_changes = index_changes[:-1]
                for index_change in index_changes:
                    mark_map[i][j][index_change] = 1
        mark_map = np.ascontiguousarray(mark_map)
        return torch.from_numpy(mark_map.reshape((datas.shape[0], datas.shape[2], datas.shape[1]))).float()
        
        