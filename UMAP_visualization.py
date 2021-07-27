import umap
import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import sys

training_set = sys.argv[1]
protein_train = pd.read_csv(training_set)
xtrain = protein_train.iloc[:, 1:1028]
print(protein_train.iloc[:,0:3])
ytrain = protein_train.iloc[:, -1]
embedding = umap.UMAP().fit_transform(xtrain, ytrain)
fig = plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0],embedding[:, 1],s=20, c= ytrain, cmap='Set3' ,alpha=1.0)
plt.legend(loc="best",markscale=2.,numpoints=2,scatterpoints=2,fontsize=12)
plt.title('Training set embedded by supervised UMAP ',fontsize=20)
plt.savefig('DeepPPISPXGB Embedded via UMAP with Labels')
plt.show()
