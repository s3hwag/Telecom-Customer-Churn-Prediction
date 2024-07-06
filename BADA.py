
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('/Users/sehwagvijay/Desktop/dataset_DA2BA.xlsx', engine='openpyxl')
stats = data.describe()
print(stats)

def detect_outliers(df):
    outlier_indices = []
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = list(set(outlier_indices))
    return outlier_indices

outliers = detect_outliers(data)
print("Number of outliers:", len(outliers))
print("Outliers at indices:", outliers)

def detect_missing_and_zeros(df):
    missing_values = df.isnull().sum()
    zero_values = (df == 0).sum()
    return missing_values, zero_values

missing_values, zero_values = detect_missing_and_zeros(data)
print("Missing values in each column:\n", missing_values)
print("Zero values in each column:\n", zero_values)

def replace_zeros(df, column_list):
    for column in column_list:
        median = df[column].median()
        df[column] = df[column].replace(0, median)
    return df

columns_with_zeros = ['DataUsage', 'CustServCalls', 'DayMins', 'DayCalls', 'OverageFee', 'RoamMins']
data = replace_zeros(data, columns_with_zeros)

def remove_outliers(df, exclude_columns):
    clean_df = df.copy()
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        if column not in exclude_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            clean_df = clean_df[(clean_df[column] >= lower_bound) & (clean_df[column] <= upper_bound)]
    return clean_df

exclude_outlier_columns = ['Churn', 'ContractRenewal', 'DataPlan']
clean_data = remove_outliers(data, exclude_outlier_columns)

clean_data.to_excel('/Users/sehwagvijay/Desktop/cleaned_dataset.xlsx', index=False)

