import pandas as pd
from datetime import datetime, timedelta
import pytz
import math

from sklearn.preprocessing import StandardScaler

tzinfo = pytz.timezone("Europe/Paris")


def preprocess(csv_path, cols_2_norm, standard_bool=False):
    df = pd.read_csv(csv_path)

    # Preprocessing

    df = df.drop(columns='name')  # useless column in the csv
    df = df.dropna().reset_index(drop=True)  # delete the empty rows

    # converting the unix time to "human-readable" values
    df['date'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[0])  # the date
    df['yday'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[1])  # number of days since the beginning of the year
    df['total_sec'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[2])  # number of seconds since midnight

    # cyclical encoding of the yday and total_sec values thx to trigonometry
    df['cyclic_yday_cos'] = df['yday'].apply(lambda x: math.cos(x / 365 * 2 * math.pi))
    df['cyclic_yday_sin'] = df['yday'].apply(lambda x: math.sin(x / 365 * 2 * math.pi))

    df['cyclic_sec_cos'] = df['total_sec'].apply(lambda x: math.cos(x / 86400 * 2 * math.pi))
    df['cyclic_sec_sin'] = df['total_sec'].apply(lambda x: math.sin(x / 86400 * 2 * math.pi))

    # correcting the T_Depart_PV value because of an "error" in the creation of the csv
    df['T_Depart_PV'] = df['T_Depart_PV'].apply(lambda x: 40 if x > 40 else x)

    # Fit scalers
    scalers = {}
    # norm_df = df.copy()
    # if standard_bool:
    #     for x in df.columns:
    #         if x in train_cols:
    #             scaler = StandardScaler().set_output(transform="pandas")
    #             norm_df[x] = scaler.fit_transform(df[x].values.reshape(-1, 1))
    for col in cols_2_norm:
        min = df[col].min()
        max = df[col].max()
        df[col] = df[col].apply(lambda x: (x-min) / (max-min))

    min_out = df['T_Depart_PV'].min()
    max_out = df['T_Depart_PV'].max()
    df['T_Depart_PV'] = df['T_Depart_PV'].apply(lambda x: (x - min_out) / (max_out - min_out))

    return df, min_out, max_out


def tstamp_to_mydate(timestamp):
    timestamp = timestamp * 10 ** -9
    ts = datetime.fromtimestamp(timestamp, tz=tzinfo).strftime('%Y-%m-%d %H:%M:%S')
    yday = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timetuple().tm_yday

    hour = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timetuple().tm_hour
    minu = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timetuple().tm_min
    sec = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timetuple().tm_sec
    tdelta = timedelta(hours=hour, minutes=minu, seconds=sec)
    total_sec = tdelta.total_seconds()

    return ts, yday, total_sec
