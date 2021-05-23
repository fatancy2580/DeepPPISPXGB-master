#-*- encoding:utf-8 -*-
import pickle
import numpy as np
import torch
import sys
import torch as t
import torch.utils.data.sampler as sampler
from xgboost import XGBClassifier
from torch.autograd import Variable
from generator import data_generator
from models.deep_ppi import DeepPPI
from utils.config import DefaultConfig
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score


def getnewmodel(models,seq, dssp, pssm, local_features):
    models.eval()
    shapes = seq.data.shape 
    features = seq.view(shapes[0], shapes[1] * shapes[2] * shapes[3])
    features = models.seq_layers(features)  
    features = features.view(shapes[0], shapes[1], shapes[2], shapes[3])  

    features = t.cat((features, dssp, pssm), 3) 
    features = models.multi_CNN(features)  
    features = t.cat((features, local_features), 1)
    return features


def trainfunction(train_loader,newmodel):
    feature_batch = []
    label_batch = []
    for batch_idx, (seq_data, pssm_data, dssp_data, local_data, label) in enumerate(train_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                seq_var = Variable(seq_data.cuda().float())
                pssm_var = Variable(pssm_data.cuda().float())
                dssp_var = Variable(dssp_data.cuda().float())
                local_var = Variable(local_data.cuda().float())
                target_var = Variable(label.cuda().float())

            else:
                seq_var = Variable(seq_data.float())
                pssm_var = Variable(pssm_data.float())
                dssp_var = Variable(dssp_data.float())
                local_var = Variable(local_data.float())
                target_var = Variable(label.float())  # 就是标签
                print(target_var)

        output = getnewmodel(newmodel,seq_var, dssp_var, pssm_var, local_var)
        feature_batch.append(output.data.cpu().numpy())
        label_batch.append(label.tolist())

    feature_data = np.concatenate(feature_batch, axis=0)
    label_data = np.concatenate(label_batch, axis=0)
    return feature_data,label_data


def constructXGBoost(x_train, y_train, x_valid):
    XGB = XGBClassifier(learning_rate =0.07,
                         n_estimators=393,
                         max_depth=5,
                         min_child_weight=1,
                         subsample=0.61,
                         colsample_bytree=0.4,
                         nthread=6,
                         scale_pos_weight=5.6,
                         reg_alpha=0,
                         reg_lambda=1.51,
                         tree_method='gpu_hist',
                         seed=27
                         )
    XGB = XGB.fit(x_train, y_train)
    predict_pos_proba = XGB.predict_proba(x_valid)[:, 1]
    predict_label = XGB.predict(x_valid)
    return predict_pos_proba, predict_label

def plotcrossvalidationROCurve(x_train, y_train):
    fpr_sum = []
    tpr_sum = []
    recall_sum = []
    mean_fpr = np.linspace(0, 1, 1000)
    mean_recall = np.linspace(0, 1, 10000)
    AUROC_sum = []
    i = 1
    cv = StratifiedKFold(n_splits=10)
    for train, valid in cv.split(x_train, y_train):
        predict_pos_proba, predict_label = constructXGBoost(x_train[train], y_train[train], x_train[valid])
        fpr, tpr, _ = roc_curve(y_train[valid], predict_pos_proba)
        roc = auc(fpr, tpr)
        AUROC = (round(roc, 3))
        tpr_sum.append(np.interp(mean_fpr, fpr, tpr))  
        tpr_sum[-1][0] = 0.0  
        AUROC_sum.append(AUROC)
        plt.plot(fpr, tpr, lw=1, alpha=0.7, label='ROC fold %d (AUROC = %0.3f)' % (i, AUROC))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tpr_sum, axis=0)
    mean_tpr[-1] = 1.0  
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(AUROC_sum)  
    plt.plot(mean_fpr, mean_tpr, color='lawngreen',
             label=r'Mean ROC (AUROC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8) 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves:10-fold cross-validation')
    plt.legend(loc="lower right", prop={"size": 6})
    plt.savefig('10-fold_ROC_Curves')
    plt.show()


if __name__ == '__main__':
    path_dir = "/home/xyj/Project/DeepPPISP-master/checkpoints/deep_ppi_saved_models"
    model_file = "{0}/DeepPPI_model_epoch1_train1.dat".format(path_dir)
    train_data = ["dset186","dset164","dset72"]

    class_nums = 1
    window_size = 3
    ratio = (2, 1)
    batch_size = 32
    configs = DefaultConfig()
    model = DeepPPI(class_nums, window_size, ratio)
    model.load_state_dict(torch.load(model_file))
    newmodel = model.cuda()
    train_sequences_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_sequence_data.pkl'.format(key) for key in train_data]
    train_dssp_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_dssp_data.pkl'.format(key) for key in train_data]
    train_pssm_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_pssm_data.pkl'.format(key) for key in train_data]
    train_label_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_label.pkl'.format(key) for key in train_data]
    all_list_file = '/home/xyj/Project/DeepPPISP-master/data_cache/all_dset_list.pkl'
    test_list_file = '/home/xyj/Project/DeepPPISP-master/data_cache/testing_list.pkl'



    with open('/home/xyj/Project/DeepPPISP-master/indexes/train_index_epoch1_train1.pkl', "rb") as ti:
        train_index = pickle.load(ti)

    with open(test_list_file, "rb") as fp:
        test_list = pickle.load(fp)

    train_samples = sampler.SubsetRandomSampler(train_index)
    train_dataSet = data_generator.dataSet(window_size, train_sequences_file, train_pssm_file, train_dssp_file,
                                       train_label_file,
                                       all_list_file)
    train_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=batch_size,
                                               sampler=train_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    feature_data, label_data = trainfunction(train_loader,newmodel)
    plotcrossvalidationROCurve(feature_data, label_data)
		
