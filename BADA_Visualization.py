
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('/Users/sehwagvijay/Desktop/cleaned_dataset.xlsx', engine='openpyxl')
stats = data.describe()
print(stats)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='RoamMins', y='OverageFee', data=data, hue='Churn')
plt.title('Scatter Plot: Roaming Minutes vs. Overage Fee')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['DataUsage'], bins=20, kde=True)
plt.title('Histogram: Data Usage')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='CustServCalls', data=data)
plt.title('Box Plot: Customer Service Calls by Churn')
plt.show()

plt.figure(figsize=(10, 6))
corr = data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heat Map: Correlation Matrix')
plt.show()

