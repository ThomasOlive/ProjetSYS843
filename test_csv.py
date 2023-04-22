import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Export_21-01-2021_au_17-03-2023.csv")

print(df['T_Ext_PV'].max())
print(df['T_Ext_PV'].min())
print(df['T_RDC_PV'].max())
print(df['T_RDC_PV'].min())

plt.hist(df['T_Ext_PV'])
plt.xlabel('T_Ext_PV')
plt.ylabel('Frequency')
plt.show()