# Verify the importance of global features


from main import constructXGBoost,calculateEvaluationMetrics
import pandas as pd
import matplotlib.pyplot as plt

def plot_roc_prc_curce(x_train, y_train,  x_test, y_test,local_xtrain,local_ytrain,local_xtest,local_ytest):

    sn, sp, prec, acc, F1, mcc, AUROC, AUPRC, fpr, tpr, precision, recall = calculateEvaluationMetrics(x_train, y_train,
                                                                                                       x_test, y_test)

    sn_v, sp_v, prec_v, acc_v, F1_v, mcc_v, AUROC_v, AUPRC_v, fpr_v, tpr_v, precision_v, recall_v = calculateEvaluationMetrics(
        local_xtrain,local_ytrain,local_xtest,local_ytest)  #

    fig1 = plt.figure(figsize=(5, 4)) 
    ax1 = fig1.add_subplot(111, title='ROC curve',ylabel='True Positive Rate', xlabel='False Positive Rate')
    ax1.plot(fpr, tpr, color='lawngreen', label='Global & local features AUROC = %0.3f' % AUROC)
    ax1.plot(fpr_v, tpr_v, color='deepskyblue', label='local features AUROC = %0.3f' % AUROC_v)
    plt.xlim(0,1)
    plt.ylim(0,1)
    ax1.legend(loc='lower right',fontsize=7)
    ax1.plot([0, 1], [0, 1], 'r--')
    plt.savefig('/home/xyj/Project/DeepPPISPXGB/graph/global_local_AUROC.pdf')

    fig2 = plt.figure(figsize=(5, 4))  
    ax2 = fig2.add_subplot(111, title='Precision-Recall curve', ylabel='precision', xlabel="recall")
    ax2.plot(recall, precision, color='lawngreen', label='Global & local features AUPRC = %0.3f' % AUPRC)
    ax2.plot(recall_v, precision_v, color='deepskyblue', label='local features AUPRC = %0.3f' % AUPRC_v)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax2.legend(loc='lower right', fontsize=7)
    ax2.plot([0, 1], [1, 0], 'r--')
    plt.savefig('/home/xyj/Project/DeepPPISPXGB/graph/global_local_AUPRC.pdf')



if __name__=='__main__':
    global_train = pd.read_csv(r'/home/xyj/Project/DeepPPISP-master/get_data/second_train_new_epoch1_train1.csv')
    xtrain = global_train.iloc[:, 1:1028]  
    ytrain = global_train.iloc[:, -1]
    global_test = pd.read_csv(r'/home/xyj/Project/DeepPPISP-master/get_data/second_test_new_epoch1_train1.csv')
    xtest = global_test.iloc[:, 1:1028]  
    ytest = global_test.iloc[:, -1]

    local_train = pd.read_csv(r'/home/xyj/Project/DeepPPISP-master/get_data/local_feature_train.csv')
    local_xtrain = local_train.iloc[:, 1:344]  
    local_ytrain = local_train.iloc[:, -1]
    local_test = pd.read_csv(r'/home/xyj/Project/DeepPPISP-master/get_data/local_feature_test.csv')
    local_xtest = local_test.iloc[:, 1:344]  
    local_ytest = local_test.iloc[:, -1]

    plot_roc_prc_curce(xtrain,ytrain,xtest,ytest,local_xtrain,local_ytrain,local_xtest,local_ytest)


