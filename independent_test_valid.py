#-*- encoding:utf-8 -*-

import pickle
import numpy as np
import torch
import sys
import torch as t
import torch.utils.data.sampler as sampler
from xgboost import XGBClassifier
from prettytable import PrettyTable
from torch.autograd import Variable
from generator import data_generator
from models.deep_ppi import DeepPPI
from utils.config import DefaultConfig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score

def getnewmodel(model,seq, dssp, pssm, local_features):
    model.eval()
    shapes = seq.data.shape  # shape指的是序列的shape[32,1,500,20]
    features = seq.view(shapes[0], shapes[1] * shapes[2] * shapes[3])
    features = model.seq_layers(features)  # 将稀疏矩阵转为密集矩阵，放入全连接层
    features = features.view(shapes[0], shapes[1], shapes[2], shapes[3])  # 再把序列的特征转为四维，因为pssm和dssp是四维的

    features = t.cat((features, dssp, pssm), 3)  # 将序列和其他两个特征进行拼接
    features = model.multi_CNN(features)  # 得到的特征是二维的
    features = t.cat((features, local_features), 1)
    return features
def trainfunction(train_loader,model):
    model.eval()
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


        output = getnewmodel(model,seq_var, dssp_var, pssm_var, local_var)
        feature_batch.append(output.data.cpu().numpy())
        label_batch.append(label.tolist())

    feature_data = np.concatenate(feature_batch, axis=0)
    print(feature_data.shape)
    label_data = np.concatenate(label_batch, axis=0)
    return feature_data,label_data

def constructXGBoost(x_train, y_train, x_test):
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
                         tree_method= 'gpu_hist',
                         seed=27
                         ) # objective='binary:logistic',
    XGB = XGB.fit(x_train, y_train)
    predict_pos_proba = XGB.predict_proba(x_test)[:, 1]
    predict_label = XGB.predict(x_test)
    return predict_pos_proba, predict_label

def calculateEvaluationMetrics(x_train, y_train,x_test,y_test,pos_label=1):
    predict_pos_proba, predict_label = constructXGBoost(x_train, y_train, x_test)
    pos_num = np.sum(y_test == pos_label)
    neg_num = y_test.shape[0] - pos_num
    tp = np.sum((y_test == pos_label) & (predict_label == pos_label))
    tn = np.sum(y_test == predict_label) - tp
    sn = (format(tp / pos_num, '.3f'))
    sp = (format(tn / neg_num, '.3f'))
    acc = (format((tp + tn) / (pos_num + neg_num), '.3f'))
    fn = pos_num - tp
    fp = neg_num - tn
    prec = (format(tp / (tp + fp), '.3f'))
    prec_nofloat = tp / (tp + fp)
    rec_nofloat = tp / (tp + fn)
    F_1 = 2 * prec_nofloat * rec_nofloat / (prec_nofloat + rec_nofloat)
    F1 = (round(F_1, 3))
    mcc = (format((tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))), '.3f'))
    PRC = average_precision_score(y_test, predict_pos_proba)
    fpr, tpr, _ = roc_curve(y_test, predict_pos_proba)
    roc = auc(fpr, tpr)
    AUPRC = (round(PRC, 3))
    AUROC = (round(roc, 3))
    precision, recall, thresholds = precision_recall_curve(y_test, predict_pos_proba)
    return sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall



def plotAUPRCandAUROCcurce_table(x_train, y_train, x_test, y_test,valid_feature,valid_label):
    sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(x_train, y_train,x_test,y_test)
    sn_v, sp_v, prec_v, acc_v, F1_v, mcc_v, AUROC_v, AUPRC_v, fpr_v, tpr_v, precision_v, recall_v  = calculateEvaluationMetrics(x_train, y_train,valid_feature,valid_label)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, xlabel="recall", ylabel='precision', title="Precision-Recall curve")
    ax1.plot(recall, precision, color='lawngreen', label='AUPRC_test = %0.3f' % AUPRC)
    ax1.plot(recall_v, precision_v, color='deepskyblue', label='AUPRC_valid = %0.3f' % AUPRC_v)
    ax1.legend(loc='lower right')
    ax1.plot([0, 1], [1, 0], 'r--')

    ax2 = fig.add_subplot(122, title='ROC curve', ylabel='True Positive Rate', xlabel='False Positive Rate')
    ax2.plot(fpr, tpr, color='lawngreen', label='AUR0C_test = %0.3f' % AUROC)
    ax2.plot(fpr_v, tpr_v, color='deepskyblue', label='AUR0C_valid = %0.3f' % AUROC_v)
    ax2.legend(loc='lower right')
    ax2.plot([0, 1], [0, 1], 'r--')
    plt.show()
    #绘制表格
    x = PrettyTable(["validation_mode", "ACC", "Precision", "Recall", "F1", "MCC", "Specificity", "AUROC", "AUPRC"])
    x.add_row(["independence_test", acc, prec, sn, F1, mcc, sp, AUROC, AUPRC])
    x.add_row(["independence_valid", acc_v, prec_v, sn_v, F1_v, mcc_v, sp_v, AUROC_v, AUPRC_v])
    print(x)




if __name__ == '__main__':

    path_dir = "/home/xyj/Project/DeepPPISP-master/checkpoints/deep_ppi_saved_models"
    #需要修改
    model_file = "{0}/DeepPPI_model_epoch1_train1.dat".format(path_dir)
    train_data = ["dset186","dset164","dset72"]

    class_nums = 1
    window_size = 3
    ratio = (2, 1)
    batch_size = 32
    configs = DefaultConfig()
    model = DeepPPI(class_nums, window_size, ratio)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    newmodel = model.cuda()
    train_sequences_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_sequence_data.pkl'.format(key) for key in train_data]
    train_dssp_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_dssp_data.pkl'.format(key) for key in train_data]
    train_pssm_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_pssm_data.pkl'.format(key) for key in train_data]
    train_label_file = ['/home/xyj/Project/DeepPPISP-master/data_cache/{0}_label.pkl'.format(key) for key in train_data]
    all_list_file = '/home/xyj/Project/DeepPPISP-master/data_cache/all_dset_list.pkl'
    test_list_file = '/home/xyj/Project/DeepPPISP-master/data_cache/testing_list.pkl'


    #加载训练样本
    with open('/home/xyj/Project/DeepPPISP-master/indexes/train_index_epoch1_train1.pkl', "rb") as ti:
        train_index = pickle.load(ti)
    with open('/home/xyj/Project/DeepPPISP-master/indexes/eval_index_epoch1_train1.pkl', "rb") as ei:
        eval_index = pickle.load(ei)
    with open(test_list_file, "rb") as fp:
        test_list = pickle.load(fp)


    train_samples = sampler.SubsetRandomSampler(train_index)
    valid_samples = sampler.SubsetRandomSampler(eval_index)
    test_samples = sampler.SubsetRandomSampler(test_list)



    train_dataSet = data_generator.dataSet(window_size, train_sequences_file, train_pssm_file, train_dssp_file,
                                       train_label_file,
                                       all_list_file)
    train_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=batch_size,
                                               sampler=train_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    test_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=batch_size,
                                               sampler=test_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=batch_size,
                                               sampler=valid_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)

    train_feature, train_label = trainfunction(train_loader,newmodel)
    test_feature, test_label = trainfunction(test_loader,newmodel)
    print(test_feature[0])
    valid_feature,valid_label= trainfunction(valid_loader,newmodel)
    plotAUPRCandAUROCcurce_table(train_feature, train_label, test_feature, test_label,valid_feature,valid_label)









