# Customer Churn Prediction and Retention Analytics

## 1. Abstract
This project analyses customer records and develops a machine learning classifier to identify customers who may leave a service. The solution combines data preparation, exploratory analysis, model evaluation, and a simple prediction interface.

## 2. Problem Statement
Customer churn can reduce recurring revenue and increase the cost of acquiring replacement customers. The objective is to identify patterns associated with churn and build a model that supports customer-retention decisions.

## 3. Objectives
- Inspect and clean the supplied dataset.
- Explore customer characteristics and churn patterns.
- Compare Logistic Regression and Random Forest.
- Evaluate models using precision, recall, F1-score, and ROC-AUC where available.
- Present predictions through a simple Streamlit application.
- Communicate findings through charts and a dashboard.

## 4. Dataset
Dataset file: [enter the actual filename].  
Number of records and columns: [copy from run output].  
Target column: [Churn or Exited].  
Describe the actual fields after inspecting the dataset.

## 5. Tools and Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, and Power BI (if dashboard is completed).

## 6. Methodology
1. Data loading and inspection.
2. Cleaning blank values, handling missing values, and removing duplicate records.
3. Exploratory data analysis and visualisation.
4. Train-test split with stratification.
5. Numeric imputation/scaling and categorical imputation/one-hot encoding.
6. Training Logistic Regression and Random Forest.
7. Comparing test-set metrics and saving the best model by F1-score.
8. Creating a prediction interface.

## 7. Results
Paste the actual values from `reports/model_metrics.csv`. Do not invent or promise a particular score. Explain the trade-off between precision and recall.

## 8. Business Recommendations
Use the observed charts and model results to identify customer groups that may benefit from retention outreach. Recommendations should be based on the actual results, not assumptions. A model prediction indicates risk, not certainty or causation.

## 9. Limitations
- The model is trained on historical data and may not generalise to all customers.
- Prediction quality depends on data quality and the selected target.
- The model does not prove why a customer leaves.
- Retention actions should be tested and monitored.

## 10. Future Improvements
- Add a Power BI dashboard.
- Tune model hyperparameters and compare additional classifiers.
- Monitor model drift and evaluate performance on newer data.
- Test retention strategies using controlled experiments.

## 11. Conclusion
Summarise the actual patterns, model results, and practical implications after running the project.
