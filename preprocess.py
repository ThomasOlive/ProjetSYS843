from pandas.core.api import DataFrame
import pandas as pd
from datetime import datetime, timedelta
import pytz
import math

from sklearn.preprocessing import StandardScaler

tzinfo = pytz.timezone("Europe/Paris")


def preprocess(csv_path, train_cols, standard_bool=False):
    df = pd.read_csv(csv_path)

    # Preprocessing

    df = df.drop(columns='name')
    df = df.dropna().reset_index(drop=True)

    df['date'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[0])
    df['yday'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[1])
    df['total_sec'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[2])

    df['cyclic_yday_cos'] = df['yday'].apply(lambda x: math.cos(x / 365 * 2 * math.pi))
    df['cyclic_yday_sin'] = df['yday'].apply(lambda x: math.sin(x / 365 * 2 * math.pi))

    df['cyclic_sec_cos'] = df['total_sec'].apply(lambda x: math.cos(x / 86400 * 2 * math.pi))
    df['cyclic_sec_sin'] = df['total_sec'].apply(lambda x: math.sin(x / 86400 * 2 * math.pi))

    df.loc[df['T_Depart_PV'] >= 40]['T_Depart_PV'] = 40

    # Fit scalers
    scalers = {}
    norm_df = df.copy()
    if standard_bool:
        for x in df.columns:
            if x in train_cols:
                scaler = StandardScaler().set_output(transform="pandas")
                norm_df[x] = scaler.fit_transform(df[x].values.reshape(-1, 1))
    # df = df.reset_index()
    return norm_df


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
