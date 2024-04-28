import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_excel("SPY.xlsx", sheet_name="Sheet1")
df["Percent Change"] = df["Total Return Index (Gross Dividends)"].pct_change()
df = df[["Percent Change"]]
df.dropna(inplace=True)

plt.hist(df['Percent Change'], bins=50, edgecolor='black', range=(-0.05, 0.05))
plt.xlabel('Percent Change')
plt.ylabel('Frequency')
plt.title('Histogram of Percent Change')
plt.show()