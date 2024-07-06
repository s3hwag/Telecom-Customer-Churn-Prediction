from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_excel('/Users/sehwagvijay/Desktop/Telecom-Customer-Churn-Prediction/cleaned_dataset.xlsx', engine='openpyxl')

reduced_data = data.drop(['MonthlyCharge'], axis=1)
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1])

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
forest_predictions = random_forest_model.predict(X_test)
forest_auc = roc_auc_score(y_test, random_forest_model.predict_proba(X_test)[:, 1])

print("Logistic Regression Classification Report:")
print(classification_report(y_test, logistic_predictions))
print(f"Logistic Regression ROC-AUC: {logistic_auc:.2f}")

print("Random Forest Classification Report:")
print(classification_report(y_test, forest_predictions))
print(f"Random Forest ROC-AUC: {forest_auc:.2f}")

new_customer = np.array([[120, 1, 1, 5.0, 2, 250.0, 100,40, 10.0, 12.0]])

forest_churn_prediction = random_forest_model.predict(new_customer)
forest_churn_probability = random_forest_model.predict_proba(new_customer)[:, 1]

print("Random Forest Prediction (0 = No Churn, 1 = Churn):", forest_churn_prediction[0])
print("Random Forest Churn Probability:", forest_churn_probability[0])
