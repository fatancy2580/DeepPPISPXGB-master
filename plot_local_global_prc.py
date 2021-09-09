#The PR curves of the local features and the hybrid features.
# The green curve is PR curves of local features and the blue curve is PR curve of hybrid features.
# The red dotted line is a control line on which AUPRC = 0.5.

from main import calculateEvaluationMetrics
import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_roc_curce(x_train, y_train,  x_test, y_test,local_xtrain,local_ytrain,local_xtest,local_ytest):

    sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(x_train, y_train,
                                                                                                       x_test, y_test)

    sn_v, sp_v, prec_v, acc_v, F1_v, mcc_v, AUROC_v, AUPRC_v, fpr_v, tpr_v, precision_v, recall_v = calculateEvaluationMetrics(
        local_xtrain,local_ytrain,local_xtest,local_ytest)


    plt.figure(figsize=(5, 4))  # 创建画布    print("交叉验证:", acc_v)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall curve")
    plt.plot(recall, precision, color='lawngreen', label='Global & local features AUPRC = %0.3f' % AUPRC)
    plt.plot(recall_v, precision_v, color='deepskyblue', label='local features AUPRC = %0.3f' % AUPRC_v)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend(loc='lower right',fontsize=7)
    plt.plot([0, 1], [1, 0], 'r--')
    plt.savefig('global_local_AUPRC.pdf')
    plt.show()


if __name__=='__main__':
    global_train_path = sys.argv[1]
    global_test_path = sys.argv[2]
    local_train_path = sys.argv[3]
    local_test_path = sys.argv[4]
    global_train = pd.read_csv(global_train_path)
    xtrain = global_train.iloc[:, 1:-1]
    ytrain = global_train.iloc[:, -1]
    global_test = pd.read_csv(global_test_path)
    xtest = global_test.iloc[:, 1:-1]
    ytest = global_test.iloc[:, -1]

    local_train = pd.read_csv(local_train_path)
    local_xtrain = local_train.iloc[:, 1:-1]
    local_ytrain = local_train.iloc[:, -1]
    local_test = pd.read_csv( local_test_path)
    local_xtest = local_test.iloc[:, 1:-1]
    local_ytest = local_test.iloc[:, -1]

    plot_roc_curce(xtrain,ytrain,xtest,ytest,local_xtrain,local_ytrain,local_xtest,local_ytest)


