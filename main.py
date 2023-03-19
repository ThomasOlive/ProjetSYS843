# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from datetime import datetime, timedelta
import pytz
from preprocess import *
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM


# df = preprocess("Export_test_appel.csv")
# df = preprocess("Export_21-01-2021_au_17-03-2023.csv")
df = preprocess("Export_26-01-2023_au_22-02-2023.csv")


print(df)
#
df.plot(x='date', y='T_RDC_PV')
df.plot(x='date', y='T_Ext_PV')
plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
