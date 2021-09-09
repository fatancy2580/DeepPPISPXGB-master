import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.metrics import auc,roc_curve


def loadXGBoost(weight_path,x_test):
    wh = open(weight_path, 'rb')
    xgb = pickle.load(wh)
    score = xgb.predict_proba(x_test)
    predict_pos_proba = score[:, 1]
    return predict_pos_proba

def calculateEvaluationMetrics(pre_path,raw_path, x_test_r,y_test_r,x_test_p,y_test_p):
    predict_pos_proba_r = loadXGBoost(raw_path,x_test_r)
    fpr, tpr, _ = roc_curve(y_test_r, predict_pos_proba_r)
    roc_r = auc(fpr, tpr)
    AUROC_r = (round(roc_r, 3))

    predict_pos_proba_p = loadXGBoost(pre_path,x_test_p)
    fpr_p, tpr_p, _ = roc_curve(y_test_p, predict_pos_proba_p)
    roc_p = auc(fpr_p, tpr_p)
    AUROC_p = (round(roc_p, 3))

    fig1 = plt.figure(figsize=(5, 4))
    ax1 = fig1.add_subplot(111, title='ROC curve',xlabel='False Positive Rate',ylabel='True Positive Rate')
    ax1.plot(fpr, tpr, color='lawngreen', label='Raw features(AUROC:%0.3f)' % AUROC_r)
    ax1.plot(fpr_p, tpr_p, color='deepskyblue', label=' Preprocessing features(AUROC:%0.3f)' % AUROC_p)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax1.legend(loc='lower right', fontsize=7)
    ax1.plot([0, 1], [0, 1], 'r--')
    plt.savefig(r'Raw and preprocessing features.pdf')





if __name__ == '__main__':
    test_data_r_path = sys.argv[1]  # Raw feature test data
    test_data_p_path = sys.argv[2]  # Preprocessing feature test data
    pre_path = sys.argv[3] # Preprocessing feature training model,Pre_feature_XGB.pickle
    raw_path = sys.argv[4] # Raw feature training model,Raw_feature_XGB.pickle

    test_data_r = open(test_data_r_path)
    test_data_p = open(test_data_p_path)


    test_data_r = pd.read_csv(test_data_r)
    test_data_p = pd.read_csv(test_data_p)



    x_test = test_data_r.iloc[:,:-1]
    y_test = test_data_r.iloc[:,-1]

    x_test_p = test_data_p.iloc[:,1:-1]
    y_test_p  = test_data_p.iloc[:,-1]


    calculateEvaluationMetrics(pre_path,raw_path,
                               x_test.values,y_test.values.ravel(),
                               x_test_p,y_test_p)













