# Software Defect Prediction: Comparative Analysis of TabNet vs Traditional ML Models

## Overview

This repository presents a comprehensive comparative analysis of **TabNet** and traditional machine learning models such as **XGBoost, Random Forest, SVM, Gradient Boosting, and Logistic Regression** for software defect prediction. The study evaluates model performance based on accuracy, precision, recall, F1-score, and ROC-AUC to determine the most effective approach for enhancing software quality.

## Features

- **Data Preprocessing:** 
  - Handling missing values
  - Feature scaling using `StandardScaler`
  - Addressing class imbalance with `SMOTE`
- **Exploratory Data Analysis (EDA):**
  - Class distribution visualization
  - Correlation heatmaps
  - Outlier detection with boxplots
- **Model Training and Evaluation:**
  - Individual models: XGBoost, Random Forest, SVM, Gradient Boosting, Logistic Regression, and TabNet
  - Ensemble models: Voting and Stacking classifiers
  - Performance evaluation using accuracy, precision, recall, F1-score, and confusion matrices
- **Hyperparameter Optimization:** 
  - Automated tuning using `Optuna` for TabNet
- **Visualization:**
  - Feature importance plots
  - Model performance comparison charts
  - ROC curves for performance analysis

## Dataset

The analysis is conducted using the **Unified Bug Dataset**, which contains various software metrics and bug occurrence data. The dataset undergoes rigorous preprocessing and feature selection techniques to enhance prediction accuracy.

## Installation

To set up the project environment, run the following command:

```bash
pip install -r requirements.txt
