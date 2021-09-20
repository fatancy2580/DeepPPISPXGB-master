# The ROC curves of the local features and the hybrid features.
#The green curve is ROC curves of local features and the blue curve is ROC curve of hybrid features.
#The red dotted line is a control line on which AUROC = 0.5.

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score

def constructXGBoost(x_train, y_train, x_test):
    XGB = XGBClassifier(learning_rate =0.07,
                         n_estimators=393,
                         max_depth=5,
                         min_child_weight=1,
                         objective='binary:logistic',
                         subsample=0.61,
                         colsample_bytree=0.4,
                         nthread=6,
                         scale_pos_weight=5.6,
                         reg_alpha=0,
                         reg_lambda=1.51,
                         seed=27
                         )
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
    tp = np.array(tp, dtype=np.float64)
    tn = np.array(tn, dtype=np.float64)
    fp = np.array(fp, dtype=np.float64)
    fn = np.array(fn, dtype=np.float64)
    mcc = (format((tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))), '.3f'))
    PRC = average_precision_score(y_test, predict_pos_proba)
    fpr, tpr, _ = roc_curve(y_test, predict_pos_proba)
    roc = auc(fpr, tpr)
    AUPRC = (round(PRC, 3))
    AUROC = (round(roc, 3))
    precision, recall, thresholds = precision_recall_curve(y_test, predict_pos_proba)
    return sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall

def plot_roc_curce(x_train, y_train,  x_test, y_test,local_xtrain,local_ytrain,local_xtest,local_ytest):

    sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(x_train, y_train,
                                                                                                       x_test, y_test)

    sn_v, sp_v, prec_v, acc_v, F1_v, mcc_v, AUROC_v, AUPRC_v, fpr_v, tpr_v, precision_v, recall_v = calculateEvaluationMetrics(
        local_xtrain,local_ytrain,local_xtest,local_ytest)  #

    fig1 = plt.figure(figsize=(5, 4))  # 创建画布    print("交叉验证:", acc_v)
    ax1 = fig1.add_subplot(111, title='ROC curve',ylabel='True Positive Rate', xlabel='False Positive Rate')
    ax1.plot(fpr, tpr, color='lawngreen', label='Global & local features AUROC = %0.3f' % AUROC)
    ax1.plot(fpr_v, tpr_v, color='deepskyblue', label='local features AUROC = %0.3f' % AUROC_v)
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax1.legend(loc='lower right',fontsize=7)
    ax1.plot([0, 1], [0, 1], 'r--')
    plt.savefig('/home/xyj/Project/DeepPPISPXGB/graph/global_local_AUROC.pdf')

    fig2 = plt.figure(figsize=(5, 4))  # 创建画布    print("交叉验证:", acc_v)
    ax2 = fig2.add_subplot(111, title='Precision-Recall curve', ylabel='precision', xlabel="recall")
    ax2.plot(recall, precision, color='lawngreen', label='Global & local features AUPRC = %0.3f' % AUPRC)
    ax2.plot(recall_v, precision_v, color='deepskyblue', label='local features AUPRC = %0.3f' % AUPRC_v)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax2.legend(loc='lower right', fontsize=7)
    ax2.plot([0, 1], [1, 0], 'r--')
    plt.savefig('global_local_AUPRC.pdf')



if __name__=='__main__':
    global_train_path = sys.argv[1] # global&local_feature_testing_set
    global_test_path = sys.argv[2] # global&local_feature_training_set
    local_train_path = sys.argv[3] # local_feature_training_set
    local_test_path = sys.argv[4] # local_feature_testing_set
    global_train = pd.read_csv(global_train_path)
    xtrain = global_train.iloc[:, 1:-1]
    ytrain = global_train.iloc[:, -1]
    global_test = pd.read_csv(global_test_path)
    xtest = global_test.iloc[:, 1:-1]
    ytest = global_test.iloc[:, -1]

    local_train = pd.read_csv(local_train_path)
    local_xtrain = local_train.iloc[:, 1:-1]
    local_ytrain = local_train.iloc[:, -1]
    local_test = pd.read_csv(local_test_path)
    local_xtest = local_test.iloc[:, 1:-1]
    local_ytest = local_test.iloc[:, -1]

    plot_roc_curce(xtrain,ytrain,xtest,ytest,local_xtrain,local_ytrain,local_xtest,local_ytest)


