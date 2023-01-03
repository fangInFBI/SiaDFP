import csv
from sklearn import ensemble
from tqdm import tqdm
from sklearn import metrics
from loadData import DiskDataset
from torch.utils.data import Dataset,DataLoader
import os
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from rgf.sklearn import RGFClassifier
from sklearn.svm import SVC
    
def model_initial(model):
    if model == 'BAYES':
        return GaussianNB()
    if model == 'DT':
        return DecisionTreeClassifier()
    if model == 'GBDT':
        return ensemble.GradientBoostingClassifier()
    if model == 'RF':
        return ensemble.RandomForestClassifier(n_estimators=3000,max_depth = 6, criterion = "entropy")
    if model == 'RGF':
        return RGFClassifier(max_leaf=3000,
                    algorithm="RGF_Sib",
                    test_interval=100,
                    verbose=True)
    if model == 'SVM':
        return SVC(kernel = 'poly', degree = 13)

if __name__ == '__main__':

    model = 'RGF' # BAYES, DT(DecisionTree), GBDT(GradientBoostingDecisionTree), RGF, RF(RandomForest), SVM

    data_len = 25
    pre_len = 7
    batch_size = 128
    test_batch_size = 128
    save_version = model+'_'+str(data_len)+'_'+str(pre_len)

    #dataset path
    data_path = '../dataset/dataset-1'
    train_data_path = data_path + 'train_data.npy'
    train_label_path = data_path + 'train_label.npy'
    test_data_path = data_path + 'test_data.npy'
    test_label_path = data_path + 'test_label.npy'


    #mk result file
    result_path = './result/'
    result_file = model+'.csv'
    result_heads = ['model', 'name', 'TP', 'FP', 'TN', 'FN', 'precision', 'accuracy', 'recall', 'F1']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path+result_file, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(result_heads)
        f.close()


    print('load train data')
    train_dataset = DiskDataset(train_data_path, train_label_path)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
    val_dataset = DiskDataset(test_data_path, test_label_path)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle= True, num_workers = 8, batch_size = test_batch_size)

    #build model
    MLmodel = model_initial(model)


    print('training')
    print(len(train_dataloader))
    for i ,datas in tqdm(enumerate(train_dataloader)):
        data, label = datas
        m,n,p = data.size()
        data = data.reshape(m,-1)
        MLmodel.fit(data,label)

    pickle.dump(MLmodel,open('./model/'+model+'.h5','wb'))


    pre = []
    groudTrue = []
    print('testing')
    print(len(val_dataloader))
    for i ,datas in enumerate(val_dataloader):
        data, label = datas
        data = data.reshape(data.shape[0], -1)
        predict = MLmodel.predict(data)
        pre.extend(predict)
        groudTrue.extend(label)
    confm = metrics.confusion_matrix(groudTrue,pre)

    with open(result_path + result_file,'a') as f:
        csv_write = csv.writer(f)
        result = [model, save_version, confm[0,0], confm[1,0],confm[1,1],confm[0,1],metrics.precision_score(groudTrue,pre, pos_label=0), metrics.accuracy_score(groudTrue,pre, pos_label = 0),metrics.recall_score(groudTrue,pre, pos_label = 0),metrics.f1_score(groudTrue,pre, pos_label = 0)]
        csv_write.writerow(result)
        f.close()
    print('saving')
    print('Finished!')