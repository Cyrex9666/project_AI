# Machine learning stock performance classifier project

Through the construction of a walk-forward machine learning pipeline, I tested whether technical indicators can predict the direction of a stock, compared a naive majority-class baseline to logistic regression, random forest and gradient boosting models.


# Project Motivation

This project explores the supervised learning application of ML in the context of 
the stock market, an area of finance I am personally highly interested in

This project aims to expand the breadth and depth of my knowledge of how the theories from the classroom are used in real world applications

# Index Analysed

 QQQ

## Project Pipeline
```text
Raw OHLCV data
      ↓
Data cleaning
      ↓
Feature engineering
      ↓
Target creation
      ↓
Walk-forward train/test splits
      ↓
Model comparison
      ↓
Evaluation using accuracy, macro precision, macro recall, macro F1
      ↓
Confusion matrix and feature importance analysis