# Exploratory Data Analysis

This notebook contains code for performing exploratory data analysis (EDA) on the dataset. It includes visualizations and summary statistics to understand the data better.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("../data/processed/your_processed_data.csv")

# Display the first few rows of the dataset
data.head()
```


```python
# Summary statistics
data.describe()
```


```python
# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(x="your_column_name", data=data)
plt.title("Count of Your Column")
plt.show()
```
