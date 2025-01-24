# Run script in Spyder

import pandas as pd

my_df = pd.DataFrame({"Height" : [1.98, 1.77, 1.76, 1.80, 1.64],
                      "Weight" : [99, 81, 70, 86, 82]})

# Standardization
from sklearn.preprocessing import StandardScaler

scale_standard = StandardScaler()
scale_standard.fit_transform(my_df)
my_df_standardized = pd.DataFrame(scale_standard.fit_transform(my_df), columns = my_df.columns)


# Normalization
from sklearn.preprocessing import MinMaxScaler

scale_norm = MinMaxScaler()
scale_norm.fit_transform(my_df)
my_df_normalized = pd.DataFrame(scale_norm.fit_transform(my_df), columns = my_df.columns)