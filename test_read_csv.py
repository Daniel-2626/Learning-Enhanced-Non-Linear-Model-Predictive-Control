import pandas as pd
import numpy as np

df = pd.read_csv("current_intp1.csv", names=["current"])
arr = df.to_numpy(dtype=np.float32)
print(df.head())
print(arr[0].item())