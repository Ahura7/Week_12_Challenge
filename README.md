**Week_12_Challenge**
USYD FinTech Week 12 Challenge
Coded in Jupyter Lab using Python Scripts

# Module 12 Report

## Overview of the Analysis

### Purpose
This report focuses on the analysis of credit risk poses a classification problem that’s inherently imbalanced in a set of loan data. In this dataset the healthy loans easily outnumber risky loans. This analysis aims to use and compare logistic regression machine learning models before and after oversampling adjustments to determine prediction capacity of each model.

### Data
The dataset includes labelled historical loan data with a flag/column identifying loan status.
* In this dataset the imbalance can be easily seen as healthy loans (~75000) out number high-risk loans (~2500) 30:1

### Approach
In this analysis:
* The data was split into Training and Testing Sets
* A Logistic Regression model was then created with the original imbalanced data
* Subsequently, a predictition using  Logistic Regression Model with resampled training data was also performed

**LogisticRegression:** In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.
**RandomOverSampler:**  An object to over-sample the minority class(es) by picking samples at random with replacement. 

The accuracy of each approach above was assessed using a balanced accuracy score, confustion matrix and a classification report.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: **LogisticRegression of imbalanced data**
  * Description of Model 1 Accuracy, Precision, and Recall scores.
    * Balanced Accuracy Score: 0.9520479254722232
    * Confusion Matrix: array([[18663,   102],
                               [   56,   563]])
    * Precision: 100% for healthy loans and 85% for high-risk loans
    * Recall Score: 99% for healthy loans and 90% for high-risk loans


* Machine Learning Model 2: **LogisticRegression of oversampled data**
  * Description of Model 2 Accuracy, Precision, and Recall scores.
    * Balanced Accuracy Score: 0.9936781215845847 *improved accuracy*
    * Confusion Matrix: array([[18649,   116],
                               [    4,   615]]) *reduced false positive/negatives*
    * Precision: 100% for healthy loans and 84% for high-risk loans
    * Recall Score: 99% for healthy loans and 99% for high-risk loans (*improved over the previous approach*)

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* The logistic regression model fits the oversampled data somewhat better with an increased accuracy score and reduced false positive/negative identification. This model should be prioritised over the non oversampled data, however, to assess all the option fully, it is recommended that an under sampled method be tested also.
* The performance of the model is impacted by the large 30:1 imbalance in the data as we try ti predict the occurances of high-risk loans



# Resources
* Resources discussed in week 12 modules were used in this exercise
* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
* https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
* https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html


