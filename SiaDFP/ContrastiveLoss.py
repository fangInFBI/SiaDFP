# Custom Contrastive Loss

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.stats import wasserstein_distance
import pytwed

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin, loss_type, metric):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_type = loss_type
        self.metric = metric
        self.maxLoss = torch.tensor(10000)
        

    def forward(self, output1, output2, label):
        if self.metric == 'Euclide':
            distance = F.pairwise_distance(output1, output2)
        else:
            distance = self.Distance(output1, output2)
            
            
        
        if self.loss_type == 'margin':
            loss_contrastive = torch.mean((label.float()) * torch.pow(distance.float(), 2) +
                                      (1-label.float()) * torch.pow(torch.clamp(self.margin - distance.float(), min=0.0), 2)) 
            
        elif self.loss_type == 'tanh':
            sig_distance = torch.tanh(distance)
            loss_contrastive = torch.mean((label.float()) * torch.abs(torch.log(1-sig_distance.float())) + 
                                      (1-label.float()) * torch.abs(torch.log(sig_distance.float())))
            loss_contrastive = torch.min(loss_contrastive, self.maxLoss)
        return loss_contrastive
    
    def distance_(self,w1,w2):
        d = abs(w2 - w1)
        return d
    
    def dtw(self, s1,s2):
        m = len(s1)
        n = len(s2)

        # 构建二位dp矩阵,存储对应每个子问题的最小距离
        dp = [[0]*n for _ in range(m)] 

        # 起始条件,计算单个字符与一个序列的距离
        for i in range(m):
            dp[i][0] = self.distance_(s1[i],s2[0])
        for j in range(n):
            dp[0][j] = self.distance_(s1[0],s2[j])

        # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1]) + self.distance_(s1[i],s2[j])

        return dp[-1][-1]
    
    def Distance(self,out1, out2):
        front = out1.cpu().detach().numpy()
        later = out2.cpu().detach().numpy()
        result = np.zeros(len(front))
        for i in range(len(front)):
            if self.metric == 'DTW':
                result[i] = self.dtw(front[i], later[i])
            if self.metric == 'TWED':
                result[i] = pytwed.twed(front[i], later[i])
            if self.metric == 'Wassertein':
                result[i] = wasserstein_distance(front[i], later[i])
        return Variable(torch.from_numpy(result), requires_grad=True).cuda()
            
