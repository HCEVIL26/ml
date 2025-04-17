
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Fetch data
data = fetch_california_housing()

# Create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column to the DataFrame
df['Target'] = data.target

df.shape

print("\nSummary Statistics:")
print(df.describe())

df.hist(bins = 30, figsize = (20,15))
plt.show()

df.boxplot(['AveRooms'],figsize = (10,10))
plt.show()

correlation_matrix = df.corr()
correlation_matrix

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of california housing Features')
plt.show()

