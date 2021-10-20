
from datetime import datetime
import pyforest
import numpy as np
import pandas as pd
import time
from numpy import diag
from numpy import zeros
import os
import logging
import argparse
import configparser
import sys

from math import radians, cos, sin, asin, sqrt
def interpolate(df) :
    size = 256
    np.random.seed(0)
    index1=random.sample(range(2,255), 3)
    
    df2=df.copy()
    df2['c_w']=df['watermarked_lat']
    df2['c_w_long']=df['watermarked_long']

    for i in index1:
        t=random.uniform(0.0, 0.0002)
        a_lat = float(df.iloc[i:i+1].watermarked_lat.values[0])
        b_lat = float(df.iloc[i+1:i+2].watermarked_lat.values[0])

        a_long = df.iloc[i:i+1].watermarked_long.values[0]
        b_long = df.iloc[i+1:i+2].watermarked_long.values[0]

       
        a_time = (df.iloc[i:i+1].time.values[0])
        b_time = df.iloc[i+1:i+2].time.values[0]
        td1=(b_time - a_time)
       
        if int(td1) > 1.5e+10 or int(td1) < 0 :
            print(td1)
            continue;
        noised_lat=(b_lat - a_lat) * t + a_lat
        noised_long=(b_long - a_long) * t +  a_long
        temporal =(b_time - a_time) * t + a_time
        line = pd.DataFrame({"traj_id": df.traj_id.values[0],'time':temporal, 'latitude':0.0,'longitude':0.0, 'time_diff':0,  'cum_sum':0, 'watermarked_lat':0.0,'watermarked_long':0.0,'c_w':noised_lat,'c_w_long':noised_long}, index=[i])
        df2 = pd.concat([df2.iloc[:i], line, df2.iloc[i:]]).reset_index(drop=True)
    return df2 


def distance(x):
    y = x.shift()
    return haversine_np(x['watermarked_long'], x['watermarked_lat'], y['watermarked_long'], y['watermarked_lat']).fillna(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="execution learning")
    parser.add_argument("-c", "--configfile", default="../config.ini",
                        help="select configuration file default configuration.ini ")
    parser.add_argument("-d", "--dataset", action="store_true", default=False)
    args = parser.parse_args()

    # logging to stdout and file
    global config
    config = configparser.ConfigParser()
    # read config to know path to store log file
    config.read(args.configfile)


    i=0
    start = time.time()

    noises=['interpolate']
    directory = os.path.join("/data/watermarkingTraj/data/All_results/SVD/our_data/watermarked_data/")
    for root,dirs,files in os.walk(directory):
        for file in files:
            df=pd.read_csv("/data/watermarkingTraj/data/All_results/SVD/our_data/watermarked_data/"+file,header=0)
            df=df[['traj_id','time','latitude','longitude','time_diff','cum_sum','watermarked_lat','watermarked_long']]
            trip_id=df.traj_id.values[0]
            for noise in noises:
                pd.options.mode.chained_assignment = None
                df["time"]=df['time'].astype('datetime64[s]')
                noisy_trace = interpolate(df)
                df_noisy = pd.DataFrame(noisy_trace)
                traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'
                traj_folder = traj_folder + "/"
                final_folder=traj_folder
                final_folder = final_folder + "/"
                path1=final_folder + noise
                if not os.path.isdir(path1):
                    os.mkdir(path1)
                df_noisy.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'+noise+'/'+str(trip_id)+'.csv',index=False,header=True)
elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)


