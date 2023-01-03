os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CompareNet(nn.Module):
    def __init__(self, numOfSmart, model , data_len = 50,input_dim = 15, hidden_size = 64, gru_num_layer = 2,dropout=0.2):
        super(CompareNet,self).__init__()
        self.model = model
        
        
        self.hidden_size = hidden_size
        
        self.cnnlstm_ = 32 * int((numOfSmart - 4 + 1)/2)
        
        if 'cp' in self.model:
            self.input_channel = 2
        else:
            self.input_channel = 1
            
        if '2D-att' in model:
            hidden_size = numOfSmart
            self.w_x_q = nn.Parameter(torch.Tensor(
                hidden_size, hidden_size))
            self.w_x_k = nn.Parameter(torch.Tensor(int(data_len), 1))

            self.w_y_q = nn.Parameter(torch.Tensor(int(data_len), int(data_len)))
            self.w_y_k = nn.Parameter(torch.Tensor(1,hidden_size))

            nn.init.xavier_uniform_(self.w_x_q)
            nn.init.xavier_uniform_(self.w_x_k)

            nn.init.xavier_uniform_(self.w_y_q)
            nn.init.xavier_uniform_(self.w_y_k)
        
        if '1D-att' in self.model:
            self.w_omega = nn.Parameter(torch.Tensor(numOfSmart, numOfSmart))
            self.u_omega = nn.Parameter(torch.Tensor(1,data_len))

            nn.init.xavier_uniform_(self.w_omega)
            nn.init.xavier_uniform_(self.u_omega)
            
        if 'LSTM' in self.model:
            if 'CNN' in self.model:
                self.CNNLSTM1 = nn.Sequential(
                nn.Conv2d(self.input_channel, 32, (1,4)),
                nn.MaxPool2d((1,2))
                )
                self.CNNLSTM2 = nn.LSTM(self.cnnlstm_, 32,2)
                self.FC2 = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32*data_len,1),
                    nn.Sigmoid()
                ) 
            else:
                self.LSTM1 = nn.LSTM(numOfSmart,100,1,dropout = 0.25)
                self.LSTM2 = nn.LSTM(100,100,2)
                self.LSTM3 = nn.LSTM(100,50,1)
                self.FC =nn.Sequential( 
                    nn.Flatten(),
                    nn.Linear(50*data_len,1),
                    nn.Sigmoid()
                )
            
        if model == 'BaseAMANDA':
            hidden_size = numOfSmart
            self.classifier_pred = nn.Sequential(
                                                 nn.Linear(hidden_size, hidden_size*2),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(hidden_size*2, hidden_size),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(hidden_size, 1),
#                                                
                                                 )
            
            
            
            
        if model == 'AMANDA':
            self.first_layer = nn.Linear(numOfSmart, input_dim)

            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, dropout=dropout, num_layers=gru_num_layer, batch_first = True)

            self.w_omega = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
            self.u_omega = nn.Parameter(torch.Tensor(1,data_len))

            nn.init.xavier_uniform_(self.w_omega)
            nn.init.xavier_uniform_(self.u_omega)

            self.classifier_pred = nn.Sequential(
                                                 nn.Linear(hidden_size, hidden_size*2),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(hidden_size*2, hidden_size),
                                                 nn.LeakyReLU(),
                                                 nn.Flatten(),
                                                 nn.Linear(data_len * hidden_size, 1),
#                                                  nn.Sigmoid()
                                                 )
        if model == 'AMANDA_2D-att':
            self.first_layer = nn.Linear(numOfSmart, input_dim)

            self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, dropout=dropout, num_layers=gru_num_layer, batch_first = True)
            
            self.fearture_dim_x = numOfSmart
            
            
            self.classifier_pred = nn.Sequential(
                                                 nn.Linear(hidden_size*data_len, hidden_size*2),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(hidden_size*2, hidden_size),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(hidden_size, 1),
#                                                  nn.Sigmoid()
                                                 )
            
            
        if model == 'LPAT':
            self.LPAT1 = nn.LSTM(numOfSmart, 100,1,dropout = 0.3)
            self.LPAT2 = nn.Linear(100, 64)
            self.LPAT3 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(data_len*64, 64),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        if model == 'BaseNTAM':
            
            self.FC1 = nn.Linear(numOfSmart, 1)
            self.FC2 = nn.Linear(numOfSmart*data_len, 1)
            self.Flatten = nn.Flatten()
            self.Sigmoid = nn.Sigmoid()
            
            
        if model == 'NTAM':
            self.data_len = data_len
            self.pos_map = nn.Embedding(data_len, numOfSmart, padding_idx = 0)
            nn.init.xavier_uniform_(self.pos_map.weight)
            
            
            self.hidden_size = 100
            self.attention_heads = 2
            self.head_size = int(self.hidden_size/self.attention_heads)
            
            self.query = nn.Linear(numOfSmart, self.hidden_size)
            self.key = nn.Linear(numOfSmart, self.hidden_size)
            self.value = nn.Linear(numOfSmart, self.hidden_size)
            
            self.dropout = nn.Dropout(0.3)
            
            self.FC1 = nn.Linear(self.hidden_size, 1)
            self.FC2 = nn.Linear(self.hidden_size*data_len, 1)
            self.Flatten = nn.Flatten()
            self.Sigmoid = nn.Sigmoid()
            
            
    def forward(self, x, gradient_0 = None, gradient_1 = None):
        x = x.type(torch.FloatTensor)
        x = x.cuda()
        output = x
        
        if self.model == 'LSTM':
            output,(h,c) = self.LSTM1(output)
            output,(h,c) = self.LSTM2(output)
            output,(h,c) = self.LSTM3(output)
            output = self.FC(output)
            return output
        
        if self.model == '1D-att_LSTM':
            output, score = self.attention(output)
            output,(h,c) = self.LSTM1(output)
            output,(h,c) = self.LSTM2(output)
            output,(h,c) = self.LSTM3(output)
            output = self.FC(output)
            return output
        if self.model == '2D-att_LSTM':
            output = self.att_2D(output)
            output,(h,c) = self.LSTM1(output)
            output,(h,c) = self.LSTM2(output)
            output,(h,c) = self.LSTM3(output)
            output = self.FC(output)
            return output
            
        if self.model == 'AMANDA':
            
            output = self.first_layer(output)
            output,(h,c) = self.gru(output)
            output, score = self.attention(output)
            output = self.classifier_pred(output)
            return output
        if self.model == 'AMANDA_2D-att':
            output = self.first_layer(output)
            output,(h,c) = self.gru(output)
            output = self.att_2D(output)
            output = nn.Flatten()(output)

            output = self.classifier_pred(output)
            return output
        
        if self.model == 'LPAT':
            output,(h,c) = self.LPAT1(output)
            output = self.LPAT2(output)
            output = self.LPAT3(output)
            return output
        
        if self.model == 'NTAM':
            #Position
            idx = torch.arange(self.data_len, device = x.device)
            pos_map = self.pos_map(idx)
            output = output + pos_map
            
            #Transformer
            
            query = self.query(output)
            key = self.key(output)
            value = self.value(output)
            
            query = self.transpose_for_scores(query)
            key = self.transpose_for_scores(key)
            value = self.transpose_for_scores(value)
            
            attention_score = torch.matmul(query, key.transpose(-1,-2))
            attention_score = attention_score / math.sqrt(self.head_size)
            
            attention_score = nn.Softmax(dim = -1)(attention_score)
            
            attention_score = self.dropout(attention_score)
            
            output = torch.matmul(attention_score, value)
            output = output.permute(0,2,1,3).contiguous()
            new_shape = output.size()[:-2] + (self.hidden_size,)
            
            output = output.view(*new_shape)
            
            #Time-aware
            
            scores = F.softmax(self.FC1(output), dim = 1)
            output = torch.mul(output, scores)
            output = self.Flatten(output)
            output = self.FC2(output)
            output = self.Sigmoid(output)
            print('test')
            return output
            
            
        if 'cp' in self.model:
            CP_map = self.CPD(output)
            CP_map = CP_map.cuda()
            output = torch.stack((CP_map, output),1)
        else: 
            pass
            
        if self.model == 'CNNLSTM':
            output = output.unsqueeze(1)
            output = self.CNNLSTM1(output)
            batch, kernel,H,W = output.shape
            output = output.view(batch,H,-1)
            output,(h,w) = self.CNNLSTM2(output)
            output = self.FC2(output)
            return output
        if self.model == '1D-att_CNNLSTM':
            output, _ = self.attention(output)
            output = output.unsqueeze(1)
            output = self.CNNLSTM1(output)
            batch, kernel,H,W = output.shape
            output = output.view(batch,H,-1)
            output,(h,w) = self.CNNLSTM2(output)
            output = self.FC2(output)
            return output
        if self.model == '2D-att_CNNLSTM':
            output = self.att_2D(output)
            output = output.unsqueeze(1)
            output = self.CNNLSTM1(output)
            batch, kernel,H,W = output.shape
            output = output.view(batch,H,-1)
            output,(h,w) = self.CNNLSTM2(output)
            output = self.FC2(output)
            return output
        
        return output
    

    def CPD(self, data):
        datas = data.T.cpu().detach().numpy()
        mark_map = np.zeros((data.size()[0],data.size()[1]))
        for data in datas:
            algo = rpt.Window(width = 14, min_size = 3,jump = 1).fit(data)
            index_changes = algo.predict(n_bkps = 3)
            index_changes = index_changes[:-1]
            for index_change in index_changes:
                data[index_change] = 1
        return torch.from_numpy(datas.T)

    def attention(self, x):
        
        k = torch.matmul(self.u_omega, x)
        v = torch.matmul(x, self.w_omega)
    
        v = v.permute(0,2,1)
        score = torch.matmul(k, v)
        score  = score.permute(0,2,1)
        score = nn.functional.softmax(score, dim = 1)
        score = score.expand(x.shape[0],x.shape[1], x.shape[2])
        scored_x = x * score
        return scored_x, score
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def att_2D(self, output):
        
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
            output = output.squeeze(1)
            return output
        
        