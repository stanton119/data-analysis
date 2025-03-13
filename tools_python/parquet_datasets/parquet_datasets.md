# Exploring parquet datasets

Parquet files are a columinar data format we can use to store dataframes. They can be stored in partitions, which can allow us to load only a subset of the data. This is useful is we are filtering the data, as we can do that without loading it all into memory.

Import stuff:


```python
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
```

## Load data
Here we use the UCI heart disease data set found on Kaggle:  
`https://www.kaggle.com/ronitf/heart-disease-uci`


```python
dir_path = os.getcwd()
heart_data = pd.read_csv(os.path.join(dir_path, "data", "heart.csv"))
heart_data.sort_values(by=["age", "sex", "cp"], inplace=True)
heart_data.reset_index(inplace=True, drop=True)
heart_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>202</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>0</td>
      <td>1</td>
      <td>118</td>
      <td>210</td>
      <td>0</td>
      <td>1</td>
      <td>192</td>
      <td>0</td>
      <td>0.7</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34</td>
      <td>1</td>
      <td>3</td>
      <td>118</td>
      <td>182</td>
      <td>0</td>
      <td>0</td>
      <td>174</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>138</td>
      <td>183</td>
      <td>0</td>
      <td>1</td>
      <td>182</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>120</td>
      <td>198</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>1</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Partition data
To split the data we need an appropriate column to split by. Based on it's histogram, we can use the age column to divide the data.


```python
heart_data["age"].plot(kind="hist")
plt.xlabel("Age")
plt.show()

heart_data["partition"] = (heart_data["age"] / 10).round().astype(int)
```


    
![svg](parquet_datasets_files/parquet_datasets_5_0.svg)
    


## Save data
We save the data to parquet files twice: once normally and once with partitioning.

When saving the file normally we produce a single `heart.parquet` file. When saving with partitions we create a folder `heart_partition` as follows:
```
heart_partition
├── bracket=3
│   └── ___.parquet
├── bracket=4
│   └── ___.parquet
...
└── bracket=8
    └── ___.parquet
```

Each column we use for partitioning the dataframe splits the folder structure, and the filtered data will be saved as a parquet file inside the respective folder.


```python
# Save to parquet
heart_data.to_parquet(
    os.path.join(dir_path, "data", "heart.parquet"),
    index=False,
)

# Save to partitioned parquet
heart_data.to_parquet(
    os.path.join(dir_path, "data", "heart_partition"),
    partition_cols=["partition"],
    index=False,
)
```

## Loading and testing
Pandas also supports loading from partitioned parquet datasets. Instead of loading the `.parquet` files we can point to the directory to load the whole dataset.

We resort the data after loading the partitioned dataset. When we load we concatenate the various partitions together so the ordering may not be preserved.
The `partition` column is loaded as a categorical type, so we convert back to `int` to ensure its the same as before.


```python
# Load single parquet file
heart_data_1 = pd.read_parquet(os.path.join(dir_path, "data", "heart.parquet"))

# Load parquet dataset
heart_data_2 = pd.read_parquet(os.path.join(dir_path, "data", "heart_partition"))
heart_data_2.sort_values(by=["age", "sex", "cp"], inplace=True)
heart_data_2["partition"] = heart_data_2["partition"].astype(int)
```

Comparing the resulting dataframes to the original show that they are all identical:


```python
pd.testing.assert_frame_equal(heart_data, heart_data_1)
pd.testing.assert_frame_equal(heart_data, heart_data_2)
```

## Loading a subset of data
Usually if we wanted a subset of the data, we would load the whole dataframe and filter it in memory.
One of the benefits of using partitioned datasets is that we can load a subset of data directly from disk without needing to load all the data first.
This is possible when we are filtering on one of the partition columns. This can help reduce the memory footprint of a program.

For example, in the following we filter for the 3rd parition. In this case only that partition is loaded.


```python
# Load and then filter single parquet file
heart_data_1 = pd.read_parquet(os.path.join(dir_path, "data", "heart.parquet"))
filt = heart_data_1["partition"] == 3
heart_data_1 = heart_data_1.loc[filt, :]

# Filter and load parquet dataset
heart_data_2 = pd.read_parquet(
    os.path.join(dir_path, "data", "heart_partition"), filters=[("partition", "=", "3")]
)
heart_data_2.sort_values(by=["age", "sex", "cp"], inplace=True)
heart_data_2["partition"] = heart_data_2["partition"].astype(int)

# Results are identical
pd.testing.assert_frame_equal(heart_data_1, heart_data_2)
```
