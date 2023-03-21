import pandas as pd
from datetime import datetime, timedelta
import pytz

from sklearn.preprocessing import StandardScaler

tzinfo = pytz.timezone("Europe/Paris")


def preprocess(csv_path, train_cols):
    df = pd.read_csv(csv_path)

    # Preprocessing

    df = df.drop(columns='name')
    df = df.dropna().reset_index(drop=True)

    df['date'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[0])
    df['yday'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[1])
    df['total_sec'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[2])

    # Fit scalers
    scalers = {}
    for x in df.columns:
        if x in train_cols:
            scalers[x] = StandardScaler().fit(df[x].values.reshape(-1, 1))

    # Transform data via scalers
    norm_df = df.copy()
    for i, key in enumerate(scalers.keys()):
        norm = scalers[key].transform(norm_df.iloc[:, i].values.reshape(-1, 1))
        norm_df.iloc[:, i] = norm

    # df = df.reset_index()
    return df


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