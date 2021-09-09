# The 10-fold cross validation of training set
#The ROC curves of 10-fold cross validation. The minimum AUROC value cross validation is 0.730 at the first fold.
#The maximum value of the cross validation is 0.752 at the tenth fold. The green line represents the ROC curve of the cross validation mean. 
#The mean value of AUROC is 0.741. The red dotted line is a control line on which AUROC = 0.5.

from main import  calculateEvaluationMetrics
import numpy as np
import sys
from xgboost import XGBClassifier
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import pandas as pd
from prettytable import  PrettyTable

training_set = sys.argv[1]   # training set
protein_train = pd.read_csv(training_set)

x_train = protein_train.iloc[:,1:1028]  #Contain row index
y_train = protein_train.iloc[:,-1]
x_train = np.array(x_train)
y_train = np.array(y_train)
fpr_sum = []
tpr_sum = []
mean_fpr = np.linspace(0, 1, 1000))
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
plt.show() #
