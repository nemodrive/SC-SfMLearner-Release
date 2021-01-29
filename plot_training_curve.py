import matplotlib.pyplot as plt
import pandas as pd

file = 'progress_log_summary.csv'

df = pd.read_csv(file, sep='\t')

print(df)
#df = df.rolling(4).mean()
df.plot()
plt.show()
