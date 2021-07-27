# **DeepPPISPXGB** 
A Deep Learning and XGBoost-based Method for Predicting Protein-protein Interaction Sites 


# **installation** 


umap-learn==0.4.6

numpy==1.15.0

scikit-learn==0.23.2

xgboost==1.1.0


# **data**

local_feature_training_set.csv: Preprocessing data of feature extractor contains 65869 rows and 344 columns, and rows represent the number of samples , the first 343 columns represent feature and the last column represent label 

local_feature_testing_set.csv: Preprocessing data of feature extractor contains 11791 rows and 344 columns, and rows represent the number of samples , the first 343 columns represent feature and the last column represent label 

global&local_feature_training_set.csv: Preprocessing data of feature extractor contains 65869 rows and 1028 columns, and rows represent the number of samples , the first 1027 columns represent feature and the last column represent label 

global&local_feature_testing_set.csv: Preprocessing data of feature extractor contains 11791 rows and 1028 columns, and rows represent the number of samples , the first 1027 columns represent feature and the last column represent label 



URL: https://data.mendeley.com/drafts/9tft3vz5tm

# **Usage** 

The knowledge about protein-protein interactions is beneficial to understand cellular mechanism. Protein-protein interactions are usually determined according to their protein-protein interaction sites. Due to limitation of current techniques, it is still a challenging task to detect protein-protein interaction sites. In this article, we presented a deep learning and XGBoost-based method (called DeepPPISP-XGB) for predicting protein-protein interaction sites. The deep learning model served as feature extractor to remove redundant information of the protein sequences. The Extreme Gradient Boosting algorithm was used to construct a classifier for predicting protein-protein interaction sites. The DeepPPISP-XGB achieved area under the receiver operating characteristic curve of 0.681, a recall of 0.624 and area under the precision-recall curve of 0.339, being competitive with the state-of-the-art methods. We also validate the positive role of global features in predicting protein-protein interaction sites. 

# **The architecture of DeepPPISPXGB** 
![image](https://user-images.githubusercontent.com/52057084/125041958-eaf77180-e0cb-11eb-8080-6c4314d6e148.png)



