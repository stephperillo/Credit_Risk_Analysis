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

It is important to keep in mind that this dataset is very unbalanced. There are only a small number of high risk applications compared to low risk (347 vs. 68,470).

#### Oversampling
1. `RandomOverSampler`
- The RandomOverSampler method selects random instances in the minority class (high risk applications in this case) and adds the extra samples to the training set until the majority and minority classes are balanced.

![NRO_AS](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/NRO_Accuracy_Score.png)
![NRO_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/NRO_Confusion_Matrix.png)
![NRO_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/NRO_CR.png)

2. `SMOTE`
- SMOTE is another oversampling method, however, it selects close neighbors to the minority class and synthesizes new instances.
![SMOTE_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/SMOTE_AC.png)
![SMOTE_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/SMOTE_CM.png)
![SMOTE_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/SMOTE_CR.png)

#### Undersampling
3. `ClusterCentroids`
- Undersampling decreases the size of the majority class (low risk applications) to match the minority class. There is an inherent risk of losing important data from the majority class when using this algorithm. 
- Cluster centroids undersampling identifies clusters of the majority class, then generates centroids, synthetic data points, that represent the clusters. Next, this algorithm uses the centroids to then undersample the majority class to meet the size of the minority class.
![CC_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/CC_AC.png)
![CC_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/CC_CM.png)
![CC_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/CC_CR.png)

#### Combination Sampling
4. `SMOTEENN`
- SMOTEENN uses a combination of SMOTE (Synthetic Minority Oversampling Technique) and ENN (Edited Nearest Neighbors) algorithms.
It uses a two-step process:
  1. Oversample the minority class with SMOTE.
  2. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped. 
-This method removes some outliers from the sampled dataset.

![S_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/S_AC.png)
![S_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/S_CM.png)
![S_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/S_CR.png)

#### Using machine learning models that reduce bias
5. `BalancedRandomForestClassifier` 
> A balanced random forest randomly under-samples each bootstrap sample to balance it.[^1]
- Bootstrap aggregation is also referred to as "bagging." It combines weak learners into a strong learner. I used the Balanced Random Forest algorithm with 100 estimators in this analysis. 100 samples were randomly selected in sequence then each returned back before the next instance was selected for the sample. This means that an instance can occur more than once in the final sample.

![BRF_AC](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/BRF_Accuracy_Score.png)
![BRF_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/BRF_Confusion_Matrix.png)
![BRF_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/BRF_CR.png)

This screenshot shows the top 15 features by importance according to the BalancedRandomForest model:
![Feature_Importance](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/Feature_Importance.png)

6. `EasyEnsembleClassifier`
>The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.[^2]
- AdaBoost is Adaptive Boosting. It trains a model then evaluates it to find errors. It adapts and "learns" from the errors to determine each subsequent sample. In a sense it is conducting machine learning on itself. 
- I also used 100 estimators with this classifier. 
![EEC_AS](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/EEC_AS.png)
![EEC_CM](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/EEC_Confusion_Matrix.png)
![EEC_CR](https://github.com/stephperillo/Credit_Risk_Analysis/blob/main/Resources/EEC_CR.png)


## Summary

The Easy Ensemble model with 100 classifiers yielded the highest accuracy score (0.93) out of the six algorithms used in this analysis. The nature of the algorithm makes sense that it would be more accurate than using the other methods. 

Precision was also very low when using the other five models compared to the precision of the Easy Ensemble algorithm, which had 0.09 precision in correctly predicting high risk applications, showing that it is relatively more reliable than the other algorithms. The precision for the other models were 0.03 and 0.01. 

When comparing the recall (sensitivity) amongst the different methods, the Easy Ensemble classifier yielded the best result with 0.92. Recall is the ability of the classifier to find all the high risk applications. Alternatively, a low recall indicates a larger number of false negatives.     

F1 scores: The F1 score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0. Again the Easy Ensemble model had the highest F1 in the group with a score of 0.16, which overall is rather low. In comparison, the Balanced Random Forest method had the second best F1 score of 0.06, which is still much lower than Easy Ensemble's F1 score. 

Given these results, I would recommend using the Easy Ensemble model. 

[^1]: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html 
[^2]: https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html
