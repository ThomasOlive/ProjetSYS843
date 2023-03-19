import pandas as pd
from datetime import datetime, timedelta
import pytz

tzinfo = pytz.timezone("Europe/Paris")

def preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Preprocessing

    df = df.drop(columns='name')
    df = df.dropna().reset_index(drop=True)

    df['date'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[0])
    df['yday'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[1])
    df['total_sec'] = df['time'].apply(lambda x: tstamp_to_mydate(x)[2])

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