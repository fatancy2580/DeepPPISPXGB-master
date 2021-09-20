# The ROC curves of 5-fold cross validation on perprocessed features and raw features

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import pandas as pd
import sys


def constructXGBoost():
    XGB = XGBClassifier(learning_rate =0.07,
                         n_estimators=393,
                         max_depth=5,
                         min_child_weight=1,
                         subsample=0.61,
                         colsample_bytree=0.4,
                         objective='binary:logistic',
                         nthread=6,
                         scale_pos_weight=5.6,
                         reg_alpha=0,
                         reg_lambda=1.51,
                         seed=27
                         )
    return XGB

def raw_calculateROC(xtrain, ytrain):
    cv = StratifiedKFold(n_splits=5)
    tpr_sum = []
    AUROC_sum = []
    mean_fpr = np.linspace(0, 1, 2000)
    for train, valid in cv.split(xtrain, ytrain):
        XGB = constructXGBoost()
        trained_XGB = XGB.fit(xtrain[train], ytrain[train])
        score = trained_XGB.predict_proba(xtrain[valid])
        predict_pos_proba = score[:, 1]
        fpr, tpr, _ = roc_curve(ytrain[valid],predict_pos_proba)
        roc_r = auc(fpr, tpr)
        tpr_sum.append(np.interp(mean_fpr, fpr, tpr))
        tpr_sum[-1][0] = 0.0
        AUROC_sum.append(roc_r)

    mean_fpr_r = np.linspace(0, 1, 2000)
    mean_tpr_r = np.mean(tpr_sum, axis=0)
    mean_tpr_r[-1] = 1.0
    mean_auc_r = auc(mean_fpr_r, mean_tpr_r)
    std_auc_r = np.std(AUROC_sum)
    std_auc_r = (round(std_auc_r, 3))
    return mean_fpr_r,mean_tpr_r, mean_auc_r,std_auc_r

def pre_calculateROC(xtrain, ytrain):
    cv = StratifiedKFold(n_splits=5)
    tpr_sum_p = []
    AUROC_sum_p = []
    mean_fpr = np.linspace(0, 1, 2000)
    for train, valid in cv.split(xtrain, ytrain):
        XGB = constructXGBoost()
        trained_XGB = XGB.fit(xtrain[train], ytrain[train])
        score = trained_XGB.predict_proba(xtrain[valid])
        predict_pos_proba = score[:, 1]
        fpr, tpr, _ = roc_curve(ytrain[valid], predict_pos_proba)
        roc_p = auc(fpr, tpr)
        tpr_sum_p.append(np.interp(mean_fpr, fpr, tpr))
        tpr_sum_p[-1][0] = 0.0
        AUROC_sum_p.append(roc_p)

    mean_fpr_p = np.linspace(0, 1, 2000)
    mean_tpr_p = np.mean(tpr_sum_p, axis=0)
    mean_tpr_p[-1] = 1.0
    mean_auc_p = auc(mean_fpr_p, mean_tpr_p)
    std_auc_p = np.std(AUROC_sum_p)
    std_auc_p = (round(std_auc_p, 3))
    return mean_fpr_p, mean_tpr_p, mean_auc_p,std_auc_p


if __name__ == '__main__':
    protein_train_pre_path = sys.argv[1]
    protein_train_raw_path = sys.argv[2]

    protein_train_pre = pd.read_csv(protein_train_pre_path)
    x_train_p = protein_train_pre.iloc[:, 1:-1].values  #contain row index
    y_train_p = protein_train_pre.iloc[:, -1].values

    protein_train_raw = pd.read_csv(protein_train_raw_path)
    x_train_r = protein_train_raw.iloc[:,:-1].values
    y_train_r = protein_train_raw.iloc[:,-1].values

    mean_fpr_p, mean_tpr_p, mean_auc_p, std_auc_p = pre_calculateROC(x_train_p, y_train_p)
    mean_fpr_r, mean_tpr_r, mean_auc_r, std_auc_r = raw_calculateROC(x_train_r,y_train_r)
    plt.figure(figsize=(5,4))
    plt.plot(mean_fpr_p, mean_tpr_p, color='lawngreen',
             label=r'Raw features(AUROC = %0.3f $\pm$ %0.3f)' % (mean_auc_p, std_auc_p),
             lw=2, alpha=.8)  # 在matplotlib中$\pm$表示加减的意思

    plt.plot(mean_fpr_r, mean_tpr_r, color='deepskyblue',
             label=r'Preprocessing features(AUROC = %0.3f $\pm$ %0.3f)' % (mean_auc_r, std_auc_r),
             lw=2, alpha=.8)  #

    plt.plot([0, 1], [0, 1], linestyle='--',
             lw=2, label='Chance', color='r', alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right", prop={"size": 6})
    plt.savefig('ROCcurve.pdf')
