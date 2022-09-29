import copy
import random
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc,precision_recall_curve
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from configs import config_BridgingBLT
from dataset import BIN_Data_Encoder
from models import BridgingBLT

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def show_result(AUC_List,
        AUPR_List,
        Accuracy_List,
        Recall_List,
        Precision_List,
        Sensitivity_List,
        Specificity_List,
        F1_List):

    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    Sensitivity_mean, Sensitivity_var = np.mean(Sensitivity_List), np.var(Sensitivity_List)
    Specificity_mean, Specificity_var = np.mean(Specificity_List), np.var(Specificity_List)
    F1_mean, F1_var = np.mean(F1_List), np.var(F1_List)

    print("Final test result:")

    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Specificity(std):{:.4f}({:.4f})'.format(Specificity_mean, Specificity_var)) 
    print('Sensitivity(std):{:.4f}({:.4f})'.format(Sensitivity_mean, Sensitivity_var))
    print('F1(std):{:.4f}({:.4f})'.format(F1_mean, F1_var))

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def get_kfold_data(i, datasets, k=5):   
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset


def test(data_generator, model, gen, k=0):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (drug, protein, d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        y_pred_df = pd.DataFrame(y_pred)
        y_label_df = pd.DataFrame(y_label)

        if gen:
            y_label_df.to_csv(str(k) + '_y_label.csv', index=False)
            y_pred_df.to_csv(str(k) + '_y_pred.csv', index=False)


    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr + 0.00001)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
 
    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), \
           f1_score(y_label,outputs), sensitivity1, specificity1, \
            accuracy1, recall_score(y_label, y_pred_s), precision_score(y_label, y_pred_s)



def main():
    
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    config = config_BridgingBLT()

    params = {'batch_size': config['batch_size'],
              'shuffle': True,
              'num_workers': config['num_workers'],
              'drop_last': True}

    data_path = './dataset_k_fold/' + config['dataset'] + '.csv'
    
    all_smiles = pd.read_csv(data_path)['SMILES'].tolist()
    all_protein = pd.read_csv(data_path)['Target Sequence'].tolist()
    all_label = pd.read_csv(data_path)['Label'].tolist()
    
   # all_data = [[] for i in range(len(all_smiles))]
    all_data = []
    for num in range(len(all_smiles)):
        all_data.append([all_smiles[num], all_protein[num], all_label[num]])

    # k fold
    dataset = shuffle_dataset(all_data, SEED)
    K_Fold = 5

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable,\
     Precision_List_stable, Sensitivity_List_stable, Specificity_List_stable, F1_List_stable = [], [], [], [], [], [], [], []
    
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
        train_data, test_data= get_kfold_data(i_fold, all_data)
       
        TV_data = train_data
        TV_data_len = len(TV_data)
        valid_size = int(0.2 * TV_data_len)
        train_size = TV_data_len - valid_size
        
        train_data, valid_data = torch.utils.data.random_split(TV_data, [train_size, valid_size])
        
        training_set = BIN_Data_Encoder(train_data)
        training_generator = data.DataLoader(training_set, **params)
        
        valid_set = BIN_Data_Encoder(valid_data)
        validation_generator = data.DataLoader(valid_set, **params)

        test_set = BIN_Data_Encoder(test_data)
        testing_generator = data.DataLoader(test_set, **params)
        
        loss_history = []


        model = BridgingBLT(**config)
        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model, dim=0)
            
        opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
     
        # early stopping
        max_auc = 0
        model_max = copy.deepcopy(model)

        print('--- Go for Training ---')
        torch.backends.cudnn.benchmark = True
        for epo in range(config['epoch']):
            model.train()
            for i, (drug, protein, d, p, d_mask, p_mask, label) in enumerate(training_generator):
                score = model(d.long().to(device), p.long().to(device), d_mask.long().to(device), p_mask.long().to(device))

                label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(score))

                loss = loss_fct(n, label)
                loss_history.append(loss)

                opt.zero_grad()
                loss.backward()
                opt.step()
        
                
            # gradnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if (i % 1000 == 0):
                    print('Training at Num ' + str(i_fold + 1) + ' fold, epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                        loss.cpu().detach().numpy()))

            # every epoch test
            with torch.set_grad_enabled(False):
                auc, auprc, f1, sen, spe, acc, recall, precision = test(validation_generator, model, gen=False)
                if auc > max_auc:
                    model_max = copy.deepcopy(model)
                    max_auc = auc
                    torch.save(model, 'best_model.pth')
                    print("*" * 30 + " save best model " + "*" * 30)
       


        print('--- Go for Testing ---')
        with torch.set_grad_enabled(False):
            auc, auprc, f1, sen, spe, acc, recall, precision = test(testing_generator, model_max, gen=True, k=i_fold)
            print(
                'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , Acc: ' + str(acc) + ' , F1: ' + str(f1))
        Accuracy_List_stable.append(acc)
        AUC_List_stable.append(auc)
        AUPR_List_stable.append(auprc)
        Recall_List_stable.append(recall)
        Precision_List_stable.append(precision)
        Sensitivity_List_stable.append(sen)
        Specificity_List_stable.append(spe)
        F1_List_stable.append(f1)

 
    show_result(
        AUC_List_stable,
        AUPR_List_stable,
        Accuracy_List_stable,
        Recall_List_stable,
        Precision_List_stable,
        Sensitivity_List_stable,
        Specificity_List_stable,
        F1_List_stable)
 


main()


