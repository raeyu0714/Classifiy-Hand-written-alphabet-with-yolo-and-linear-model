import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
result_df = pd.DataFrame()
for i in range(26):
    #data = pd.read_csv('exceltry.csv')
    data = pd.read_csv(chr(i+97)+".csv")
    sample_df = data.sample(n = 801, axis = 0)
    #sample_df = sample_df.iloc[ : ]
    #print(sample_df)
    result_df = pd.concat([result_df,sample_df])
print(result_df)
result_df.to_csv('final.csv', index = False)