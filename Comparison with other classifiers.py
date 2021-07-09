import pandas as pd
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def constructClassifier():
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
    Rand = RandomForestClassifier(n_estimators=393,
                                  max_depth=5,
                                  class_weight="balanced",
                                  random_state=27)
    ETC = ExtraTreesClassifier(n_estimators=393,
                                class_weight="balanced",
                                max_depth=5,
                                random_state=27)
    tree = DecisionTreeClassifier(class_weight="balanced",
                                       max_depth=5,
                                       random_state=27)
    svm = SVC(class_weight="balanced",
              probability=True,
              random_state = 27,
              max_iter=1)
    return  XGB,Rand,ETC,tree,svm

def calculateEvaluationMetrics(x_train, y_train,x_test,y_test):
    XGB, Rand, ETC, tree, svm = constructClassifier()

    #XGB algorithm
    XGB = XGB.fit(x_train, y_train)
    predict_pos_proba_X = XGB.predict_proba(x_test)[:, 1]
    fpr_X, tpr_X, _ = roc_curve(y_test, predict_pos_proba_X)
    roc_X = auc(fpr_X, tpr_X)
    AUROC_X = (round(roc_X, 3))

    PRC_X = average_precision_score(y_test, predict_pos_proba_X)
    AUPRC_X = (round(PRC_X, 3))
    precision_X, recall_X, thresholds = precision_recall_curve(y_test, predict_pos_proba_X)



    Rand = Rand.fit(x_train, y_train)
    predict_pos_proba_R = Rand.predict_proba(x_test)[:, 1]
    fpr_R, tpr_R, _ = roc_curve(y_test, predict_pos_proba_R)
    roc_R = auc(fpr_R, tpr_R)
    AUROC_R = (round(roc_R, 3))

    PRC_R = average_precision_score(y_test, predict_pos_proba_R)
    AUPRC_R = (round(PRC_R, 3))
    precision_R, recall_R, thresholds = precision_recall_curve(y_test, predict_pos_proba_R)


    ETC = ETC.fit(x_train, y_train)
    predict_pos_proba_G = ETC.predict_proba(x_test)[:, 1]
    fpr_G, tpr_G, _ = roc_curve(y_test, predict_pos_proba_G)
    roc_G = auc(fpr_G, tpr_G)
    AUROC_G = (round(roc_G, 3))

    PRC_G = average_precision_score(y_test, predict_pos_proba_G)
    AUPRC_G = (round(PRC_G, 3))
    precision_G, recall_G, thresholds = precision_recall_curve(y_test, predict_pos_proba_G)



    tree = tree.fit(x_train, y_train)
    predict_pos_proba_T = tree.predict_proba(x_test)[:, 1]
    fpr_T, tpr_T, _ = roc_curve(y_test, predict_pos_proba_T)
    roc_T = auc(fpr_T, tpr_T)
    AUROC_T = (round(roc_T, 3))

    PRC_T = average_precision_score(y_test, predict_pos_proba_T)
    AUPRC_T = (round(PRC_T, 3))
    precision_T, recall_T, thresholds = precision_recall_curve(y_test, predict_pos_proba_T)


    svm = svm.fit(x_train, y_train)
    predict_pos_proba_S = svm.predict_proba(x_test)[:, 1]
    fpr_S, tpr_S, _ = roc_curve(y_test, predict_pos_proba_S)
    roc_S = auc(fpr_S, tpr_S)
    AUROC_S= (round(roc_S, 3))

    PRC_S = average_precision_score(y_test, predict_pos_proba_S)
    AUPRC_S = (round(PRC_S, 3))
    precision_S, recall_S, thresholds = precision_recall_curve(y_test, predict_pos_proba_S)






    #进行画AUC图
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(111, title='ROC curve',ylabel='True Positive Rate', xlabel='False Positive Rate')
    ax1.plot(fpr_X, tpr_X, color='lawngreen', label='Xgboost(AUR0C:%0.3f)' % AUROC_X)
    ax1.plot(fpr_R, tpr_R, color='cyan', label='RandomForest(AUR0C:%0.3f)' % AUROC_R)
    ax1.plot(fpr_G, tpr_G, color='gold', label=' ExtraTrees(AUR0C:%0.3f)' % AUROC_G)
    ax1.plot(fpr_T, tpr_T, color='tomato',label='DecisionTreeClassifier(AUR0C:%0.3f)' % AUROC_T)
    ax1.plot(fpr_S, tpr_S, color='dodgerblue',label='SVM(AUR0C:%0.3f)' % AUROC_S)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.plot([0, 1], [0, 1], 'r--')
    plt.savefig('/home/ubuntu/MyFiles/DeepPPISPXGB/graph/AUROC_machine.pdf')
    plt.show()

    #画prc图
    fig = plt.figure(figsize=(5, 4))
    ax2 = fig.add_subplot(111, title='Precision-Recall curve', ylabel='precision', xlabel="recall")
    ax2.plot(recall_X, precision_X, color='lawngreen', label='Xgboost(AUPRC:%0.3f)' % AUPRC_X)
    ax2.plot(recall_R, precision_R, color='cyan', label='RandomForest(AUPRC:%0.3f)' % AUPRC_R)
    ax2.plot(recall_G, precision_G, color='gold', label=' ExtraTrees(AUPRC:%0.3f)' % AUPRC_G)
    ax2.plot(recall_T, precision_T, color='tomato', label='DecisionTree(AUPRC:%0.3f)' % AUPRC_T)
    ax2.plot(recall_S, precision_S, color='dodgerblue', label='SVM(AUPRC:%0.3f)' % AUPRC_S)
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax2.legend(loc='upper right')
    ax2.plot([0, 1], [1, 0], 'r--')
    plt.savefig('/home/ubuntu/MyFiles/DeepPPISPXGB/graph/AUPRC_machine.pdf')




if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    protein_train = pd.read_csv(file1)
    xtrain = protein_train.iloc[:, 1:1028]
    ytrain = protein_train.iloc[:, -1]
    protein_test = pd.read_csv(file2)
    xtest = protein_test.iloc[:, 1:1028]
    ytest = protein_test.iloc[:, -1]
    calculateEvaluationMetrics(xtrain, ytrain,xtest,ytest)