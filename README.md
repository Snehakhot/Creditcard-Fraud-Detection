# Credit Card Fraud Detection Project

## Purpose
This project aims to develop a machine learning model to predict fraudulent credit card transactions.

## Overview
Credit card fraud is a significant concern in financial transactions. This project aims to build a classification model to predict whether a transaction is fraudulent or not. The dataset used contains transactions made by credit cards in September 2013 by European cardholders. It includes 284,807 transactions, out of which only 492 are fraudulent, making it highly unbalanced.

## Problem Statement
The objective is to develop a robust classification model capable of identifying fraudulent credit card transactions accurately. By doing so, credit card companies can prevent unauthorized charges and protect their customers from financial losses.

## Setup Instructions
To run the project, follow these steps:

1. **Download Dataset**: Obtain the 'creditcard.csv' dataset.

2. **Install Dependencies**: Ensure you have the following Python libraries installed:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - imbalanced-learn

3. **Run the Script**: Execute the provided Python script 'CreditCardFraudDetection.ipynb' to train and evaluate the model.

## Data Sources
The dataset `creditcard.csv` was used for this project. It contains the following attributes:
- **Time**: Time elapsed in seconds between each transaction and the first transaction.
- **V1 - V28**: Principal components obtained with PCA (anonymized features).
- **Amount**: Transaction amount.
- **Class**: Indicates whether the transaction is fraudulent (1) or not (0).

## Code and Models
### Code Explanation
The Python code performs the following tasks:
1. **Data loading and preprocessing**: Loads the dataset, handles missing values, converts data types, and performs oversampling to address class imbalance.
2. **Feature Engineering and Selection**: Adds new features and selects relevant features using PCA and ANOVA F-test.
3. **Model Selection and Training**: Utilizes a Random Forest Classifier for training the model and evaluates its performance using cross-validation.
4. **Hyperparameter Tuning**: Uses GridSearchCV to find the best hyperparameters for the model.
5. **Model Evaluation**: Evaluates the model's performance on test data, including accuracy, confusion matrix, and classification report.
6. **Additional Visualizations**: Includes visualizations such as correlation matrix, scatter plot, and transaction volume over time.

### Instructions for Running the Code
1. Install necessary libraries:
    ```bash
    pip install pandas numpy matplotlib scikit-learn imbalanced-learn joblib seaborn
    ```
2. Clone the repository and navigate to the project directory.
3. Ensure the `creditcard.csv` dataset is placed in the same directory.
4. Run the Python script (`credit_card_fraud_detection.ipynb`).
6. The output will include cross-validation scores, model evaluation metrics, and additional visualizations.

### Reproducing Results
You can reproduce the results by following the instructions provided above and referring to the specific output metrics mentioned in the code explanations.

## Deployment in AWS SageMaker
For deployment in AWS SageMaker, you can use the trained model (`credit_card_fraud_detection_model.pkl`). Follow AWS SageMaker documentation for deploying models and serving predictions.

---

## Project Structure
- `CreditCardFraudDetection.ipynb`: Main Python script for model building and evaluation.
- `creditcard.csv`: Dataset containing credit card transactions.
- `credit_card_fraud_detection_model.pkl`: Saved trained model.

## Contact Information
For any questions or issues, please contact [Sneha Patil] at [sneha.khot1995@gmail.com].

---
Feel free to reach out if you have any questions or need further assistance!
