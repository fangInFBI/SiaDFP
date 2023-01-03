import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from statistics import mean
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
from baseline import CompareNet
from loadData import DiskDataset

def mk_csv(heads, path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(heads)
            f.close()
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

if __name__ == '__main__':
    
    batch_size = 1024
    test_batch_size = 4096
    lr_rate = 0.01
    epochs = 500
    metric = 'Euclide'

    data_len = 50
    pre_len = 7
    models = ['1D-att_LSTM','1D-att_CNNLSTM','2D-att_LSTM', '2D-att_CNNLSTM']

    result_heads = ['model', 'name', 'threshold','TP', 'FP', 'TN', 'FN', 'precision', 'F1', 'accuracy', 'recall']

    numberOfSmart = {data_len:19}

    result_path = '../result/'+metric+'_'+str(data_len)+'_'+str(pre_len)+'_compareNet_threshold_ST-2.csv'
    mk_csv(result_heads, result_path)


    file_root = '../dataprocess/'
    data_root = 'metrics/'+metric+'/'+str(data_len)+'/'+str(pre_len)+'/'
    root = file_root + data_root
    result_root = '../checkpoint/supervise/'

    data_path = '../dataset/dataset-1'
    train_data_path = data_path + 'train_data.npy'
    train_label_path = data_path + 'train_label.npy'
    test_data_path = data_path + 'test_data.npy'
    test_label_path = data_path + 'test_label.npy'

    for model in models:


        data_version = model+'_'+str(data_len) + '_' + str(pre_len)

        train_version = '_' + str(batch_size)
        data_version = '_ST-2_'
        save_version = data_version+train_version + data_version#测试的时候保留最好的，不需要加时间了

        save_path = result_root+model+'/'

        if not os.path.exists(save_path+save_version):
            os.makedirs(save_path+save_version)

        test_model = save_path + save_version +'.pt'

        #tensorboard
        tensorboard_path = '../runs/'+model+'_'+save_version
        if not os.path.exists(tensorboard_path):
            os.mkdir(tensorboard_path)
        writer = SummaryWriter(tensorboard_path)



        print('load train data...')
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        train_dataset =  DiskDataset(train_data_path, train_label_path)
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
        val_dataset = DiskDataset(test_data_path, test_label_path)
        val_dataloader = DataLoader(dataset=val_dataset, shuffle= True, num_workers = 8, batch_size = test_batch_size)

        net = CompareNet(numberOfSmart[data_len], model,int(data_len)).cuda()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.RMSprop(net.parameters(), lr = lr_rate,alpha = 0.9)



        for epoch in tqdm(range(0, epochs[model])):
            train_loss = []
            val_loss = []
            real_labels = []
            pre_labels = []
            for i , data in enumerate(train_dataloader, 0):
                part1, part2, label = data
                part1, part2, label = Variable(part1.float()).cuda(), Variable(part2.float()).cuda(), Variable(label.float()).cuda()
                output = net(part2)
                optimizer.zero_grad()

                acc = binary_acc(output, label.unsqueeze(1))
                loss_contrastive = criterion(output, label.unsqueeze(1))
                loss_contrastive.backward()

                optimizer.step()
                if i %50 == 0 :
                    print("Epoch number {}\n Current train loss {}\n Current acc {}\n".format(epoch,loss_contrastive.item(),acc))
                    train_loss.append(loss_contrastive.item())
            if epoch%100 ==0:
                torch.save(net.state_dict(),save_path+save_version+'/'+save_version+'epoch'+str(epoch)+'.pt')
            writer.add_scalar('Loss/train', mean(train_loss), epoch)


            for i,data in enumerate(val_dataloader, 0):
                part1, part2, label = data
                part1, part2, label = Variable(part1.float()).cuda(), Variable(part2.float()).cuda(), Variable(label.float()).cuda()
                output = net(part2)
                output = output.squeeze(1)
                test_loss_contrastive = criterion(output, label)
                output = torch.round(torch.sigmoid(output))
                real_labels.extend(label.cpu().detach().numpy())
                pre_labels.extend(output.cpu().detach().numpy())
                if i %50 ==0:
                    print(model +":"+ save_version)
                    print("Epoch number {}\n Current val loss {}\n".format(epoch,test_loss_contrastive.item()))
                    val_loss.append(test_loss_contrastive.item())
            real_labels = [int(x) for x in real_labels]
            pre_labels = [int(x) for x in pre_labels]
            accuracy = accuracy_score(real_labels, pre_labels)
            f1 = f1_score(real_labels, pre_labels)
            writer.add_scalar('Loss/test', mean(val_loss), epoch)
            writer.add_scalar('Performance/accuracy',accuracy,epoch)
            writer.add_scalar('Performance/F1',f1, epoch)
        writer.close()
        torch.save(net.state_dict(), save_path+save_version+'/'+save_version+".pt")


        test_net = CompareNet(numberOfSmart[data_len], model).cuda()
        test_data = DiskDataset(test_data_path, test_label_path)
        test_dataloader = DataLoader(dataset = test_data, shuffle = True, batch_size = test_batch_size, num_workers = 8)
        for model_name in tqdm(os.listdir(save_path+save_version)):

            if '.pt' not in model_name:
                continue
            test_model = save_path+save_version+'/'+model_name
            test_net.load_state_dict(torch.load(test_model))
            real_labels = []
            pre_labels = []

            for i ,data in tqdm(enumerate(test_dataloader,0)):
                x_0, x_1, real_label = data
                output = test_net(x_1)
                output = torch.sigmoid(output)
                output = output.squeeze(1)
                pre_label = torch.round(output)
                real_labels.extend(real_label.cpu().detach().numpy())
                pre_labels.extend(pre_label.cpu().detach().numpy())
            real_labels = [int(x) for x in real_labels]
            pre_labels = [int(x) for x in pre_labels]
            confusion_ = confusion_matrix(real_labels, pre_labels,labels = [0,1])
            precision_ = precision_score(real_labels, pre_labels, pos_label = 0,average='binary')
            accuracy_ = accuracy_score(real_labels, pre_labels)
            f1_ = f1_score(real_labels, pre_labels, pos_label = 0,average='binary')
            recall_ = recall_score(real_labels, pre_labels, pos_label = 0,average='binary')
            result = [model, model_name, 0.5,confusion_[0,0], confusion_[1,0], confusion_[1,1], confusion_[0,1], precision_, f1_, accuracy_, recall_]
            with open(result_path, 'a+') as file:
                result_file = csv.writer(file)
                result_file.writerow(result)
                file.close()
            print("successful!")
    print("Finished!")
                            
                            
    
        
    
        
        
