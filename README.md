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

2. `SMOTE`

#### Unersampling
3. `ClusterCentroids`

#### Combination Sampling
4. `SMOTEENN`
SMOTEENN uses a combination of SMOTE (Synthetic Minority Oversampling Technique) and ENN (Edited Nearest Neighbors) algorithms.

#### Using machine learning models that reduce bias
5. `BalancedRandomForestClassifier` 

This screenshot shows the top 15 features by importance according to the BalancedRandomForest model:
![Feature_Importance](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/Feature_Importance.png)

6. `EasyEnsembleClassifier`

## Summary
