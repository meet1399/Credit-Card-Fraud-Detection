# Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting credit card fraud using various machine learning techniques. As fraudsters are increasing day-by-day, detecting fallacious transactions through credit cards is crucial. This project implements a combination of K-Means, Logistic Regression, and Neural Networks to tackle this issue.

## Table of Contents

1. [Abstract](#abstract)
2. [Keywords](#keywords)
3. [Introduction](#introduction)
4. [Problems and Goals](#problems-and-goals)
5. [Methodology](#methodology)
    - [Data Processing](#data-processing)
    - [Data Modeling](#data-modeling)
6. [Discussion of Results](#discussion-of-results)
7. [Conclusion](#conclusion)

## Abstract

This project aims to detect credit card fraud using machine learning algorithms. Fraudulent transactions are a significant problem in the financial sector, with fraudsters constantly evolving their tactics. To address this, we use a combination of K-Means clustering, Logistic Regression, and Neural Networks. Our approach involves data preprocessing, PCA transformation, and a hybrid technique of under-sampling and oversampling to handle the highly skewed dataset. The project is implemented in Python using Jupyter Notebooks.

## Keywords

- Machine Learning
- K-Means
- Logistic Regression
- Neural Networks
- Credit Card Fraud Detection

## Introduction

Fraud is a wrongful or criminal deception intended to result in financial or personal gain. In this system, we use two mechanisms: fraud prevention and fraud detection. Fraud prevention is a defensive strategy to prevent misrepresentation from starting. Fraud detection involves identifying and stopping fraudulent transactions.

Credit card fraud involves illicit use of credit card information for unauthorized transactions. Technological advancements have led to an increase in credit card transactions, and consequently, fraud cases. Machine learning algorithms such as K-Means, Logistic Regression, and Neural Networks can effectively detect credit card fraud.

## Problems and Goals

The primary goal is to implement three different machine learning models to classify credit card fraud with high accuracy. The main challenge is the dataset imbalance, where frauds account for only 0.172% of transactions. We aim to minimize false negatives, which are more harmful than false positives in this context.

## Methodology

### Data Processing

The dataset consists of 30 features, with 28 of them PCA transformed and anonymized. The known features are 'Time' and 'Amount'. The dataset requires careful preprocessing to handle the imbalance and prepare it for modeling.

### Data Modeling

We used three models: a fully connected neural network, K-Means, and Logistic Regression. Principal Component Analysis (PCA) was applied to improve model accuracy. 

1. **Neural Network**: PCA on oversampled data improved accuracy from 50% to 94.56%. Adjustments to layers and activation functions had minimal impact beyond this point.
2. **K-Means**: PCA reduced data dimensionality for training. The model was set with 2 clusters representing fraud and non-fraud.
3. **Logistic Regression**: Tested with three configurations: vanilla, with oversampling and scaling, and with balanced class weights.

## Discussion of Results

Logistic regression outperformed both K-Means and the neural network. The decision boundary adjustments in logistic regression improved accuracy. The neural network showed significant improvement with PCA, and K-Means performed the poorest due to its reliance on feature similarities.

## Conclusion

The project successfully demonstrates the application of machine learning models to detect credit card fraud. Logistic regression proved to be the most effective model in this study, followed by the neural network and K-Means. Future work can explore additional models and techniques to further improve accuracy.

## References

Dataset: Kaggle Credit Card Fraud Detection
Scikit-learn documentation
Python documentation
