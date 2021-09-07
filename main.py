import numpy as np
import pandas as pd
import sys
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from prettytable import PrettyTable

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



def plotAUPRCandAUROCcurce_table(x_train, y_train, x_test, y_test):
    sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(x_train, y_train,x_test,y_test)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, xlabel="recall", ylabel='precision', title="Precision-Recall curve")
    ax1.plot(recall, precision, color='lawngreen', label='AUPRC_test = %0.3f' % AUPRC)
    ax1.legend(loc='lower right')
    ax1.plot([0, 1], [1, 0], 'r--')

    ax2 = fig.add_subplot(122,  xlabel='False Positive Rate',ylabel='True Positive Rate',title='ROC curve')
    ax2.plot(fpr, tpr, color='lawngreen', label='AUR0C_test = %0.3f' % AUROC)
    ax2.legend(loc='lower right')
    ax2.plot([0, 1], [0, 1], 'r--')
    plt.savefig('AUROC&AUPRC')
    plt.show()
    
    x = PrettyTable(["validation_mode", "ACC", "Precision", "Recall", "F1", "MCC", "Specificity", "AUROC", "AUPRC"])
    x.add_row(["independence_test", acc, prec, sn, F1, mcc, sp, AUROC, AUPRC])
    print(x)

if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    protein_train = pd.read_csv(file1)
    xtrain = protein_train.iloc[:, 1:-1]
    ytrain = protein_train.iloc[:, -1]
    protein_test = pd.read_csv(file2)
    xtest = protein_test.iloc[:, 1:-1]
    ytest = protein_test.iloc[:, -1]
    plotAUPRCandAUROCcurce_table(xtrain, ytrain, xtest, ytest)
