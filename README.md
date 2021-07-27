# **DeepPPISPXGB** 
A Deep Learning and XGBoost-based Method for Predicting Protein-protein Interaction Sites 

# **system requirement** 
DeepPPISPXGB is developed under Linux environment with python 3.7.5 Recommended RAM: > 24GB.  CUDA Version: 10.1 
# **installation** 


umap-learn==0.4.6

numpy==1.15.0

scikit-learn==0.23.2

xgboost==1.1.0


# **data**

global&local_feature_train_set.csv: this CSV file contains a training set for combining global and local feature pretraining, which size is 65869*1027 

global&local_feature_test_set.csv：this CSV file contains a testing set for combining global and local feature pretraining, which size is 11791*1027

local_feature_train_set.csv：this CSV file contains a training set for local feature pretraining, which size is 65869*343

local_feature_test_set.csv: this CSV file contains a testing set for local feature pretraining, which size is 11791*343


URL: https://drive.google.com/drive/folders/1RzX6NcuTjIOmstZO47EEmA-cIoLCo3OJ?usp=sharing

# **Usage** 

Research into protein-protein interactions is help us better understand cellular function and develop drugs. Protein-protein interactions are usually identified by determining the protein-protein interaction sites. With the development of deep learning algorithms, it has become a crucial method to predict protein-protein interaction sites. In this work, we have presented a deep learning and XGBoost-based method for predicting protein-protein interaction sites, and the method named DeepPPISP-XGB. The deep learning model served as feature extractor to remove redundant information of the protein sequence that was encoded. The XGBoost classifier was used for predicting protein-protein interaction sites on the data that was preprocessed by the feature extractor. The prediction performance show that our method is superior to other state-of-the-art methods. In the independent test, the proposed method achieved area under the receiver operating characteristic curve of 0.681, recall of 0.624 and area under the precision-recall curve of 0.339. In the 10-fold cross validation, area under the receiver operating characteristic curve reached 0.741. In addition, we compared the prediction performance of the XGBoost classifier with another popular classifier on independent test. The prediction performance demonstrate the superiority of our classifier. Ultimately, we validate global features are used for improving prediction performance.

# **The architecture of DeepPPISPXGB** 
![image](https://user-images.githubusercontent.com/52057084/125041958-eaf77180-e0cb-11eb-8080-6c4314d6e148.png)



