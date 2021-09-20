# The 10-fold cross validation of global&local_feature_training_set(preprocessing features)
#The ROC curves of 10-fold cross validation. The minimum AUROC value cross validation is 0.730 at the first fold.
#The maximum value of the cross validation is 0.752 at the tenth fold. The green line represents the ROC curve of the cross validation mean. 
#The mean value of AUROC is 0.741. The red dotted line is a control line on which AUROC = 0.5.

import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from prettytable import  PrettyTable

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

if __name__ == '__main__':
    training_set = sys.argv[1]   # global&local_feature_training_set
    protein_train = pd.read_csv(training_set)
    x_train = protein_train.iloc[:,1:1028]  #contain row index
    y_train = protein_train.iloc[:,-1]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    fpr_sum = []
    tpr_sum = []
    mean_fpr = np.linspace(0, 1, 1000)
    AUROC_sum = []
    i = 1
    cv = StratifiedKFold(n_splits=10)
    for train, valid in cv.split(x_train, y_train):
        sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(x_train[train], y_train[train], x_train[valid],y_train[valid],pos_label=1)
        x = PrettyTable(["cross validation", "ACC", "Precision", "Recall", "F1", "MCC", "Specificity", "AUROC", "AUPRC"])
        x.add_row(["{}fold".format(i), acc, prec, sn, F1, mcc, sp, AUROC, AUPRC])
        print(x)  #Print the evaluation metrics for each fold cross validation
        tpr_sum.append(np.interp(mean_fpr, fpr, tpr))
        tpr_sum[-1][0] = 0.0
        AUROC_sum.append(AUROC)
        plt.plot(fpr, tpr, lw=1, alpha=0.7, label='ROC fold %d (AUROC = %0.3f)' % (i, AUROC))
        i += 1

    # Drawing the ROC curve of the 10-fold cross validation
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tpr_sum, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(AUROC_sum)
    plt.plot(mean_fpr, mean_tpr, color='lawngreen',
             label=r'Mean ROC (AUROC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves:10-fold cross-validation')
    plt.legend(loc="lower right", prop={"size": 6})
    plt.show()
