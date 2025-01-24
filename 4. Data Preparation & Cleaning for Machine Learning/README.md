# Data Preparation & Cleaning for Machine Learning

## Introduction

### Machine Learning Prep Checklist

![ml prep checklist](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig1%20-%20ml%20prep%20checklist.png)

## Working with Missing Values

### Theory - Missing Values

- Machine learning models won’t know how to process missing data, this will cause further issues down the line- We want to make sure we pass the most important information on to the model to train
- Using `pandas`, we can:
  - Find missing values
  - Drop missing values
  - Fill missing values
    - With a constant
    - With the respective column’s mean or median
- If we don’t want to use pandas, we can use `scikit-learn`’s various preprocessing techniques - one of which is called imputation
- Allows us to input/impute replacement values for those that are missing
- `SimpleImputer` – enables the static imputation of missing values, using either a constant value or a column mean, median, or even the most common value
  - Very similar to `pandas`, why use it?
    - Using `scikit-learn` techniques allows you to link together all data preparation techniques into one single pipeline – very useful
- `KNNImputer` – uses the k-nearest-neighbors algorithm to dynamically impute missing numerical values by utilizing other data-points that appear similar across other available numerical features
  - It makes the assumption that other similar looking data points will give a better estimation of what the missing value is likely to be
 
![knn ex 1](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig2%20-%20knn%20ex%201.png)

- Unlike `SimpleImputer`, `KNNImputer` does NOT rely solely on column C for its estimation of the missing value
- It plots each data point, finds the closest data points (based on k) to the unknown point, and imputes with the average

|k=4|k=2|
|:---:|:---:|
|![k4](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig3%20-%20k4.png)|![k2](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig4%20-%20k2.png)|

- The premise – closer data points are more likely to represent the unknown or missing value
- We can also give certain close points more weight
  - If we give 9 more weight than 20, then the missing value will be closer to 9
- *MOST IMPORTANT LESSON* – we want to be passing the model the most useful information for it to learn from
  - We should always consider whether missing values meet that criteria or not
  - Sometimes it doesn’t make sense to impute values  - could make ‘too big’ of an assumption about them
  - Every case is different – context will determine whether imputation or removal is better

### Missing Values with Python

- Refer to [missing_values_with _pandas.py](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/scripts_and_data/missing_values_with_pandas.py)

```python
import pandas as pd
import numpy as np

my_df = pd.DataFrame({"A" : [1, 2, 4, np.nan, 5, np.nan, 7],
                      "B" : [4, np.nan, 7, np.nan, 1, np.nan, 2]})

# Finding Missing Values with Pandas
my_df.isna()
my_df.isna().sum()

# Dropping Missing Values with Pandas
my_df.dropna()
my_df.dropna(how = "any")
my_df.dropna(how = "all")

my_df.dropna(how = "any", subset = ["A"])
my_df.dropna(how = "any", inplace = True)

# Filling Missing Values with Pandas
my_df = pd.DataFrame({"A" : [1, 2, 4, np.nan, 5, np.nan, 7],
                      "B" : [4, np.nan, 7, np.nan, 1, np.nan, 2]})

my_df.fillna(value = 100)

mean_value = my_df["A"].mean()
my_df["A"].fillna(value = mean_value)

my_df.fillna(value = my_df.mean(), inplace = True)
```

### Missing Values with `SimpleImputer`

- Refer to [missing_values_with_simpleimputer.py](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/scripts_and_data/missing_values_with_simpleimputer.py)

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

my_df = pd.DataFrame({"A" : [1, 4, 7, 10, 13],
                      "B" : [3, 6, 9, np.nan, 15],
                      "C" : [2, 5, np.nan, 11, np.nan]})

imputer = SimpleImputer()
imputer.fit(my_df)
imputer.transform(my_df)

my_df1 = imputer.transform(my_df)

imputer.fit_transform(my_df)
my_df2 = pd.DataFrame(imputer.fit_transform(my_df), columns = my_df.columns)

imputer.fit_transform(my_df[["B"]])
my_df["B"] = imputer.fit_transform(my_df[["B"]])
```

### Missing Values with `KNNImputer`

- Refer to [missing_values_with_knnimputer.py](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/scripts_and_data/missing_values_with_knnimputer.py)

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

my_df = pd.DataFrame({"A" : [1, 2, 3, 4, 5],
                      "B" : [1, 1, 3, 3, 4],
                      "C" : [1, 2, 9, np.nan, 20]})

knn_imputer = KNNImputer()
knn_imputer = KNNImputer(n_neighbors = 1)
knn_imputer = KNNImputer(n_neighbors = 2)
knn_imputer = KNNImputer(n_neighbors = 2, weights = "distance")
knn_imputer.fit_transform(my_df)

my_df1 = pd.DataFrame(knn_imputer.fit_transform(my_df), columns = my_df.columns)
```

### Theory - Categorical Variables

- **Categorical Variables**: values made up of things like names, labels, classes, or text
- ML models don’t know how to assign numerical importance to them
- You need to be careful about assigning numbers to categories as to not assign an order (see example below)

![bad ex](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig5%20-%20bad%20ex.png)

- Instead, we can use **One-Hot-Encoding**: representation of categorical variables as binary vectors

![good ex](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig6%20-%20good%20ex.png)

- Categorical data has no order or scale, we then drop original column
- We still need to consider the **Dummy Variable Trap**
  - Where input variables perfectly predict each other – creating **multicollinearity**
  - **Multicollinearity** occurs when 2 or more input variables are highly or completely correlated with each other
    - It might not affect the precision of the model, but it makes it hard to trust the statistics around how well the model is performing and how much impact each input variable is truly having
  - The solution – drop 1 of our new dummy variables

![dummy ex](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig7%20-%20dummy%20ex.png)

### Categorical Variables One-Hot-Encoding Hands-On

- Refer to [one_hot_encoding.py](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/scripts_and_data/one_hot_encoding.py)

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

X = pd.DataFrame({"input1" : [1, 2, 3, 4, 5],
                  "input2" : ["A", "A", "B", "B", "C"],
                  "input3" : ["X", "X", "X", "Y", "Y"]})

categorical_vars = ["input1", "input2"]

one_hot_encoder = OneHotEncoder(sparse = False, drop = "first")

encoder_vars_array = one_hot_encoder.fit_transform(X[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names(categorical_vars)

encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)

X_new = pd.concat([X.reset_index(drop = True), encoder_vars_df.reset_index(drop = True)], axis = 1)
X_new.drop(categorical_vars, axis = 1, inplace = True)
```

## Outliers & Feature Scaling

### Theory - Outliers

- An outlier can be any value that differs significantly from other values
- We can either:
  - Do nothing
  - Remove them
  - Replace their values
- What we do depends on the context of our task
  - Linear regression models will be more affected by outliers than a decision tree for example
 
![lin reg outlier](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig8%20-%20lin%20reg%20outlier.png)

**Outlier Detection Approach 1 - Box Plot**

![box plot](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig9%20-%20box%20plot.png)

**Outlier Detection Approach 2 - Standard Deviation**

- Outliers are more/less than 3 standard deviations of the mean

![bell curve](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig10%20-%20bell%20curve.png)

### Outliers Hands-On

- Refer to [outliers.py](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/scripts_and_data/outliers.py)

```python
import pandas as pd

my_df = pd.DataFrame({"input1" : [15, 41, 44, 47, 50, 53, 56, 59, 99],
                      "input2" : [29, 41, 44, 47, 50, 53, 56, 59, 66]})

my_df.plot(kind = "box", vert = False)

outlier_columns = ["input1", "input2"]

# Boxplot Approach
for column in outlier_columns:
    
    lower_quartile = my_df[column].quantile(0.25)
    upper_quartile = my_df[column].quantile(0.75)
    
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 1.5
    
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    my_df.drop(outliers, inplace = True)
    
    
# Standard Deviation Approach
my_df = pd.DataFrame({"input1" : [15, 41, 44, 47, 50, 53, 56, 59, 99],
                      "input2" : [29, 41, 44, 47, 50, 53, 56, 59, 66]})

for column in outlier_columns:
    mean = my_df[column].mean()
    std_dev = my_df[column].std()
    
    min_border = mean - std_dev * 3
    max_border = mean + std_dev * 3
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    my_df.drop(outliers, inplace = True)
```

### Theory - Feature Scaling

- **Feature Scaling** is where we force the values from different columns to exist on the same scale, in order to enhance the learning capabilities of the model
- The 2 most common techniques are **Standardization** and **Normalization**
- In our example, height and weight exist on different scales:

![pre feature scaling](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig11%20-%20pre%20feature%20scaling.png)

**Standardization**

- Rescales the data to have a mean of 0 and a standard deviation of 1

  ![standardization equation](https://latex.codecogs.com/svg.image?&space;x_{standardized}=\frac{(x-mean(x))}{std.deviation(x)})

|Pre-transformation|Post-transformation|
|---|---|
|![fig12](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig12%20-%20mean%20std.png)|![fig23](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig13%20-%20standardized%20table.png)|

![standardized plot](https://github.com/ndomah/1.-The-Basics/blob/main/4.%20Data%20Preparation%20%26%20Cleaning%20for%20Machine%20Learning/img/fig14%20-%20standardized%20plot.png)

**Normalization**







