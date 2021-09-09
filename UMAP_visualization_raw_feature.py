# FIGURE 4 | Training data visualization of Raw features through UMAP
# FIGURE 5 | Testing data visualization of Raw features through UMAP 


import umap
import sys
import pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns
import numpy as np
feature_path = sys.argv[1]
protein_train = pd.read_csv(feature_path)
xtrain = protein_train.iloc[:, :-1]  #No row index
ytrain = protein_train.iloc[:, -1]
ytrain = pd.DataFrame(ytrain)

dic = {0:'non-interaction sites',1:'interaction sites'}
ls = []
for index,value in ytrain.iterrows():
    arr = np.array(value)[0]
    ls.append(dic[arr])

embedding_p = umap.UMAP().fit_transform(xtrain)
fig = plt.figure(figsize=(5, 4))
print(embedding_p[:, 0].shape)
ax1 = fig.add_subplot(111, xlabel="UMAP1", ylabel='UMAP2', title="Raw features of training data set")
sns.scatterplot(embedding_p[:, 0],embedding_p[:, 1],hue=ls,palette="pastel",sizes=10)
ax1.legend(loc="lower right")
plt.savefig('Raw features of training data set.pdf')
