# Run in Spyder

import pandas as pd

my_df = pd.read_csv('feature_selection_data.csv')

correlation_matrix = my_df.corr()