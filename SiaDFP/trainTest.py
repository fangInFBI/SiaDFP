import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from statistics import mean
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
from loadData import DiskDataset
from ContrastiveLoss import ContrastiveLoss
from SiaDFP import SiameseNet_pos_att_cnn_adjust


def show_plot(iteration,loss, path, version, loss_type):
    plt.plot(iteration,loss)
    plt.savefig(path+ loss_type+ '_' + version+ '.png',format = 'png')
    plt.show()

def mk_csv(heads, path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(heads)
            f.close()

def train(save_path, save_version, train_dataloader, val_dataloader, writer, thresholds, loss_type):
    train_counter = []
    train_loss_history = [] 
    iteration_number= 0
    val_iteration_number = 0
    val_counter = []
    val_loss_history = [] 
    min_val_loss = np.inf
    
    if not os.path.exists(save_path+save_version):
            os.makedirs(save_path+save_version)
    
    for epoch in tqdm(range(0, epochs)):
        train_loss = []
        val_loss = []
        for i, data in enumerate(train_dataloader, 0):
            part1, part2, label = data
            part1, part2, label = Variable(part1.float()).cuda(), Variable(part2.float()).cuda(), Variable(label.float()).cuda()
            out1, out2 = net(part1, part2)
            optimizer.zero_grad()
            loss_contrastive = criterion(out1, out2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i %50 == 0 :
                print("Epoch number {}\n Current train loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                train_counter.append(iteration_number)
                train_loss_history.append(loss_contrastive.item())
                train_loss.append(loss_contrastive.item())
        if epoch %200 == 0:
            torch.save(net.state_dict(), save_path+save_version+'/'+save_version+'epoch'+str(epoch)+".pt")
                
        writer.add_scalar('Loss/train', mean(train_loss), epoch)
        
        count = {}
        label_types = ['true', 'pre']
        for threshold in thresholds:
            label_ty = {}
            for label_type in label_types:
                label_ty[label_type] = []
                count[threshold] = label_ty
        for i, data in enumerate(val_dataloader, 0):
            part1, part2, label = data

            part1, part2, label = Variable(part1).cuda(), Variable(part2).cuda(), Variable(label).cuda()
            out1, out2 = net(part1, part2)
            test_loss_contrastive = criterion(out1, out2, label)
            euclidean_distance = F.pairwise_distance(out1, out2)
            if loss_type == 'tanh':
                euclidean_distance = torch.tanh(euclidean_distance)
            
            
            for threshold in thresholds:
                count[threshold]['true'].extend(label.cpu().numpy())
                euclidean_distance[euclidean_distance >= threshold] = 0
                euclidean_distance[euclidean_distance < threshold] = 1
                count[threshold]['pre'].extend(euclidean_distance.int().cpu().numpy())
            if i %50 == 0 :
                print(model +":"+ save_version)
                print("Epoch number {}\n Current val loss {}\n".format(epoch,test_loss_contrastive.item()))
                val_iteration_number +=10
                val_counter.append(val_iteration_number)
                val_loss_history.append(test_loss_contrastive.item())
                val_loss.append(test_loss_contrastive.item())
        mean_val_loss = mean(val_loss)
        writer.add_scalar('Loss/test', mean_val_loss, epoch)
    
        if mean_val_loss < min_val_loss:
            torch.save(net.state_dict(), save_path+save_version+'/'+save_version+".pt")
            min_val_loss = mean_val_loss

    writer.close()
    return net

def find_thresholds(data_loader, net):
    ps_min = 0
    ps_max = 0
    ng_min = 0
    ng_max = 0
    ps_label = torch.from_numpy(np.array([1]))
    ng_label = torch.from_numpy(np.array([0]))
    for i, data in tqdm(enumerate(test_dataloader, 0)):
        x_0, x_1, real_label = data
        output1,output2 = test_net(x_0.to(device),x_1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        if real_label.equal(ps_label):
            if euclidean_distance.item() > ps_max:
                ps_max = euclidean_distance.item()
            elif euclidean_distance.item() < ps_min:
                ps_min = euclidean_distance.item()
        else:
            if euclidean_distance.item() > ng_max:
                ng_max = euclidean_distance.item()
            elif euclidean_distance.item() < ng_min:
                ng_min = euclidean_distance.item()
    thresholds = []
    ps_min = ps_min - 1
    ng_max = ng_max + 1
    if ps_min < ng_max:
        dis = (ng_max - ps_min)/10
        while ps_min < ng_max:
            ps_min = int(ps_min*100)/100
            thresholds.append(ps_min)
            ps_min += dis
    print("successful!")
    if len(thresholds) == 0:
        output_file = open('./result/output.txt', "a+")
        output_file.write(ps_min+' '+ng_max+' ' + model+ ' '+save_version)
    return thresholds

if __name__ == '__main__':

    batch_size = 4096
    test_batch_size = 4096
    lr_rate = 0.001
    epochs = 601
    drop_rate = 0.5
    hidden_size = 32
    data_year = 2017
    
    
    metric = 'Euclide'
    data_lens = [50]
    pre_lens = [10]
    models = ['cnnlstm_pos_att10_cp']
    margins = [2]
    loss_types = ['margin']
    kernel_sizes = [4]
    result_heads = ['model', 'name', 'threshold', 'TP', 'FP', 'TN', 'FN', 'Precision', 'F1','Accuracy', 'Recall'] 

    result_path = './result/'+metric+'_'+'pre_lens'+'.csv'
    mk_csv(result_heads, result_path)
    
    result_lists = []
    for model in tqdm(models):
        for data_len in tqdm(data_lens):
            for pre_len in tqdm(pre_lens):
                for kernel_size in kernel_sizes:
                    for loss_type in loss_types:
                        tanh_count = 0
                        for margin in margins:
                            tanh_count = tanh_count + 1
                            if loss_type == 'tanh' and tanh_count ==2:
                                continue
                            data_root = 'data_'+ str(data_year) + '_' + str(data_len)+'/'+'supervise/'+'data_' + str(2017) + '_'+ str(data_len)+ '_' + str(pre_len) +'/'

                            file_root = './dataprocess/'
                            data_root = 'metrics/'+metric+'/'+str(data_len)+'/'+str(pre_len)+'/'
                            root = file_root + data_root
                            result_root = './checkpoint/supervise/'

                            data_version = metric+'_'+str(data_len) + '_' + str(pre_len)+ '_' + str(kernel_size)+'_'+loss_type
                            if loss_type == 'margin':
                                train_version = '_' + str(batch_size) + '_' + str(epochs)+'_'+loss_type+str(margin)
                                thresholds = np.arange(0, margin, margin/10)
                            else:
                                train_version = '_' + str(batch_size) + '_' + str(epochs)+'_'+loss_type
                                thresholds = np.arange(0,1,0.1)
                            save_version = data_version+train_version

                            save_path = result_root+model+'/'

                            test_model = save_path + save_version +'.pt'

                            #tensorboard
                            tensorboard_path = './runs/'+model+'_'+save_version
                            if not os.path.exists(tensorboard_path):
                                os.mkdir(tensorboard_path)
                            writer = SummaryWriter(tensorboard_path)
                            numberOfSmart = {data_len:22}
                            

                            print('load training data...')
                            train_dataset = DiskDataset()
                            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
                            val_dataset = DiskDataset()
                            val_dataloader = DataLoader(dataset=val_dataset, shuffle= True, num_workers = 8, batch_size = test_batch_size)

                            net = SiameseNet_pos_att_cnn_adjust(numberOfSmart[data_len],numberOfSmart[data_len], data_len, kernel_size,model).cuda()


                            criterion = ContrastiveLoss(margin, loss_type, metric)
                            optimizer = optim.Adam(net.parameters(), lr = lr_rate)      
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                            print('start train：' + model)
                            trained_model = train(save_path, save_version, train_dataloader, val_dataloader, writer, thresholds, loss_type)
                            print("Model Saved Successfully")



                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            print("load"+ model)

                            test_net = SiameseNet_pos_att_cnn_adjust(numberOfSmart[data_len],numberOfSmart[data_len], data_len, kernel_size,model).cuda()
                            print("load test data...")
                    
                            test_data = DiskDataset(False)
                            test_dataloader = DataLoader(dataset = test_data, shuffle = True, batch_size = 2048, num_workers = 8)
                            for model_name in tqdm(os.listdir(save_path+save_version)):
                                if '.pt' not in model_name:
                                    continue

                                test_model = save_path+save_version+'/'+model_name
                                test_net.load_state_dict(torch.load(test_model))
                                print("successful!")
                                

                                print("successful!")

                                label_0 = torch.from_numpy(np.array([0]))
                                label_1 = torch.from_numpy(np.array([1]))

                                result_lists = []
                                if margin == 500:
                                    thresholds = find_thresholds(test_dataloader, test_net)
                                if len(thresholds) == 0:
                                    continue
                                print('测试模型：'+model_name)
                                for threshold in tqdm(thresholds):
                                    real_labels = []
                                    pre_labels = []
                                    for i, data in tqdm(enumerate(test_dataloader, 0)):
                                        x_0, x_1, real_label = data
                                        real_labels.extend(real_label.cpu().detach().numpy())
                                        output1,output2 = test_net(x_0.to(device),x_1.to(device))
                                        euclidean_distance = F.pairwise_distance(output1, output2)
                                        if loss_type == 'tanh':
                                            euclidean_distance = torch.tanh(euclidean_distance)
                                            
                                        euclidean_distance[euclidean_distance <= threshold] = 1
                                        euclidean_distance[euclidean_distance > threshold] = 0
                                        pre_labels.extend(euclidean_distance.cpu().detach().numpy())
                                    real_labels = [int(x) for x in real_labels]
                                    pre_labels = [int(x) for x in pre_labels]
                                    confusion_ = confusion_matrix(real_labels, pre_labels)
                                    precision_ = precision_score(real_labels, pre_labels, pos_label = 0)
                                    accuracy_ = accuracy_score(real_labels, pre_labels )
                                    f1_ = f1_score(real_labels, pre_labels ,pos_label = 0)
                                    recall_ = recall_score(real_labels, pre_labels, pos_label = 0)
                                    result_list = [model, model_name, threshold,confusion_[0,0], confusion_[1,0], confusion_[1,1], confusion_[0,1], precision_, f1_, accuracy_, recall_]
                                    result_lists.append(result_list)
                                    writer.add_scalar('Performace/Precision', precision_, threshold)
                                    writer.add_scalar('Performace/Accuray', accuracy_, threshold)
                                    writer.add_scalar('Performace/Recall', recall_, threshold)
                                    writer.add_scalar('Performace/F1', f1_, threshold)
                                print("save result...")
                                writer.close()
                                with open(result_path, 'a+') as file:
                                    result_file = csv.writer(file)
                                    result_file.writerows(result_lists)
                                    file.close()
                                print("successful!")
    print("Finished")

