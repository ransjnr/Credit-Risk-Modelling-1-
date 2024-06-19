import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

credit_risk = pd.read_csv("UCI_credit_card.csv")
print(credit_risk.head())

# make a copy of the original
df = credit_risk.copy()

df.drop(["ID"], axis=1, inplace = True)
# check the statistics of data
print(df.describe())
# checking for missing values
print(df.isnull().sum())

# Display information about the dataset
print(df.info())

# visualization of 'LIMIT_BAL'
plt.figure(figsize=(10, 6))
sns.histplot(df['LIMIT_BAL'], bins=30, kde=True)
plt.title('Distribution of Credit Limit (LIMIT_BAL)')
plt.xlabel('Credit Limit')
plt.ylabel('Frequency')
plt.show()

# Relationship between 'AGE' and 'LIMIT_BAL'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AGE', y='LIMIT_BAL', data=df)
plt.title('Age vs Credit Limit')
plt.xlabel('Age')
plt.ylabel('Credit Limit')
plt.show()
