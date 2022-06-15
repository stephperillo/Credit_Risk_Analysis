# Credit_Risk_Analysis

## Overview
I have been tasked with using machine learning to determine credit card risk for this project.
I used different techniques to train and evaluate models with unbalanced classes, since good loans outnumber risky loans.
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I analyzed credit risk using the following steps:

1. Oversample the data using the `RandomOverSampler` and `SMOTE` (Synthetic Minority Oversampling Technique) algorithms

2. Undersample the data using the `ClusterCentroids` algorithm.

3. Use the `SMOTEENN` algorithm, a combinatorial approach of over- and undersampling.

4. Compare two machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. 

## Results

#### Oversampling
1. `RandomOverSampler`

![NRO_AS](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/NRO_Accuracy_Score.png)
![NRO_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/NRO_Confusion_Matrix.png)
![NRO_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/NRO_CR.png)

2. `SMOTE`

![SMOTE_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/SMOTE_AC.png)
![SMOTE_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/SMOTE_CM.png)
![SMOTE_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/SMOTE_CR.png)

#### Unersampling
3. `ClusterCentroids`

![CC_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/CC_AC.png)
![CC_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/CC_CM.png)
![CC_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/CC_CR.png)

#### Combination Sampling
4. `SMOTEENN`
SMOTEENN uses a combination of SMOTE (Synthetic Minority Oversampling Technique) and ENN (Edited Nearest Neighbors) algorithms.


![S_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/S_AC.png)
![S_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/S_CM.png)
![S_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/S_CR.png)

#### Using machine learning models that reduce bias
5. `BalancedRandomForestClassifier` 

![BRF_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/BRF_Accuracy_Score.png)
![BRF_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/BRF_Confusion_Matrix.png)
![BRF_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/BRF_CR.png)

This screenshot shows the top 15 features by importance according to the BalancedRandomForest model:
![Feature_Importance](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/Feature_Importance.png)

6. `EasyEnsembleClassifier`

![EEC_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/EEC_AS.png)
![EEC_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/EEC_Confusion_Matrix.png)
![EEC_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/EEC_Classification_Report.png)


## Summary
