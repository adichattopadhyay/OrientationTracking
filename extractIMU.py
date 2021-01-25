import pandas as pd
import os 

df = pd.read_csv(os.getcwd() + "/Data/aimu6.csv")
df = df.drop(df.columns[[0,5,6,7,11,12,13,14,15,16]], axis=1)

unixTime = df['unixTimes'].to_list()
time = [unixTime[i] - unixTime[0] for i in range(len(unixTime))]

df.insert(1, "time", time, True)
df = df.drop(df.columns[[0]], axis=1)

df.to_csv(os.getcwd() + '/Data/imu6.csv')