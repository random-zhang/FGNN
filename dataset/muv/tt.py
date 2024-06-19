import numpy as np
import pandas as pd
df=pd.read_csv('train.csv')
name=df.columns.tolist()
for n in name[1:]:
    print(np.nansum(df[n]))