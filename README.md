# Credit Scoring Model

## Project Overview

This project aims to build a machine learning model to predict credit default based on various personal and loan-related features. Credit scoring is crucial for financial institutions to assess the risk associated with lending money. By predicting whether a loan applicant is likely to default, we can help reduce financial losses.

## Goal

The primary goal of this project is to develop and compare different classification models to accurately identify individuals who are likely to default on their loans.

## Dataset

We use the `credit_risk_dataset.csv` file, which contains information about loan applicants, including their age, income, employment length, loan amount, interest rate, and whether they ultimately defaulted on their loan (`loan_status`).

## Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading & Initial Exploration:** Loaded the dataset and performed an initial check of its structure, data types, and identified missing values and potential outliers.
2.  **Data Preprocessing:**
    *   Handled an outlier in `person_emp_length` by capping it at 60 years.
    *   Imputed missing values in `person_emp_length` and `loan_int_rate` using the median of their respective columns.
3.  **Feature Engineering:**
    *   Converted categorical features (like `person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`) into numerical format using one-hot encoding.
    *   Visualized the distribution of the target variable (`loan_status`) to understand class balance.
4.  **Handling Class Imbalance & Data Splitting:**
    *   Split the dataset into training and testing sets.
    *   Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data to address the class imbalance (where 'no default' samples significantly outnumbered 'default' samples), ensuring models don't get biased towards the majority class.
    *   Scaled numerical features using `StandardScaler` to bring them to a similar range, which is important for many machine learning algorithms.
5.  **Model Training & Evaluation:**
    We trained and evaluated three popular classification algorithms:
    *   **Logistic Regression:** A linear model that serves as a good baseline.
    *   **Decision Tree Classifier:** A tree-based model that makes decisions based on a series of rules. We optimized its parameters using `GridSearchCV`.
    *   **Random Forest Classifier:** An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
    
    Each model was evaluated using key metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrices. We also used **ROC AUC (Receiver Operating Characteristic - Area Under the Curve)** for a comprehensive comparison of model performance.

## Key Findings

After comparing the models, the **Random Forest Classifier** emerged as the best-performing model, demonstrating a strong balance in predicting credit default. It achieved the highest F1-Score and AUC score, indicating its effectiveness in distinguishing between defaulters and non-defaulters.

## Technologies Used

*   Python
*   Pandas (for data manipulation)
*   NumPy (for numerical operations)
*   Scikit-learn (for machine learning models and utilities)
*   Imblearn-learn (for handling class imbalance)
*   Matplotlib & Seaborn (for data visualization)

## Conclusion

This project successfully implemented a machine learning pipeline to predict individual credit worthiness based on the credit_risk_dataset. After comprehensive data loading, cleaning, feature engineering (including one-hot encoding for categorical variables and addressing class imbalance with SMOTE), and feature scaling, three classification models—Logistic Regression, Decision Tree Classifier, and Random Forest Classifier—were trained and evaluated.

The models were assessed using key metrics such as Accuracy, Precision, Recall, F1-Score, and ROC AUC. Among the three, the Random Forest Classifier emerged as the best-performing model, demonstrating the highest F1-Score of 0.8005 and an impressive AUC score of 0.9274. This indicates that the Random Forest model is most effective in identifying credit risks with a good balance between precision (minimizing false positives) and recall (minimizing false negatives).

While Logistic Regression provided a baseline, and the Decision Tree performed reasonably well after hyperparameter tuning, the ensemble nature of the Random Forest allowed it to capture more complex patterns in the data, leading to superior predictive power for credit default. This model can therefore be confidently used to assess an individual's credit worthiness, helping financial institutions make more informed lending decisions.
