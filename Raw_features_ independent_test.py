# Calculate performance of raw feature independent test 
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import sys
from xgboost import XGBClassifier
from sklearn.metrics import auc,roc_curve,precision_recall_curve,average_precision_score


def constructXGBoost(x_train, y_train, x_test):
    XGB = XGBClassifier(learning_rate =0.07,
                         n_estimators=393,
                         max_depth=5,
                         min_child_weight=1,
                         subsample=0.61,
                         colsample_bytree=0.4,
                         nthread=-1,
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
    y_test = y_test.reshape(-1)
    pos_num = np.sum(y_test == pos_label)
    neg_num = y_test.shape[0] - pos_num
    tp = np.sum((y_test == pos_label)&(predict_label == pos_label))
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
    x = PrettyTable(["validation_mode", "ACC", "Precision", "Recall", "F1", "MCC", "Specificity", "AUROC", "AUPRC"])
    x.add_row(["independence_test", acc, prec, sn, F1, mcc, sp, AUROC, AUPRC])
    print(x)


if __name__ == '__main__':
    train_data_sys = sys.argv[1] #raw_feature_training_set
    test_data_sys = sys.argv[2]  #raw_feature_testing_set
    train_data = pd.read_csv(train_data_sys)
    test_data = pd.read_csv(test_data_sys)
    x_trian = np.array(train_data.iloc[:,:-1])
    y_train = np.array(train_data.iloc[:,-1]).ravel()
    x_test = np.array(test_data.iloc[:,:-1])
    y_test = np.array(test_data.iloc[:,-1]).ravel()
    plotAUPRCandAUROCcurce_table(x_trian,y_train,x_test,y_test)













