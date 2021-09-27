import random
import numpy as np
from datetime import datetime
import pandas as pd
import time
from numpy import diag
from numpy import zeros

import sys

from math import radians, cos, sin, asin, sqrt

def cropfromlast(df):
    last_value_removed=3
    df_2=df.copy()
    len1=len(df.index)
    index=len1-last_value_removed
    df_2['c_w']=df['watermarked_lat'].copy()
    df_2['c_w_long']=df['watermarked_long'].copy()
    n=len(df_2.index)
    df_2=df_2.head((n-last_value_removed))
    
    new_data = pd.DataFrame(df_2[-1:].values,index=[n-3], columns=df_2.columns)
    df_2 = df_2.append(new_data)
    new_data = pd.DataFrame(df_2[-1:].values,index=[n-2], columns=df_2.columns)
    df_2 = df_2.append(new_data)
    new_data = pd.DataFrame(df_2[-1:].values,index=[n-1], columns=df_2.columns)
    df_2 = df_2.append(new_data)

    return df_2

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
    noises=['cropfromlast']
    directory = os.path.join("/data/dadwal/watermarkingTraj/data/All_results/TrajGuard/our_data/our_data_watermarked_trajGuard_withmorebits/")

    for root,dirs,files in os.walk(directory):
        for file in files:
            df=pd.read_csv("/data/dadwal/watermarkingTraj/data/All_results/TrajGuard/our_data/our_data_watermarked_trajGuard_withmorebits/"+file,header=None)
            df.columns = ['trip_id','time','watermarked_lat', 'watermarked_long']
            df=df[['watermarked_lat', 'watermarked_long','trip_id','time']]
            trip_id=df.trip_id.values[0]
            for noise in noises:
                noisy_trace = cropfromlast(df)
                df_noisy = pd.DataFrame(noisy_trace)
                traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'
                traj_folder = traj_folder + "/"
                final_folder=traj_folder
                final_folder = final_folder + "/"
                path1=final_folder + noise
                if not os.path.isdir(path1):
                    os.mkdir(path1)

                df_noisy['trip_id']=trip_id
                df_noisy=df_noisy[['trip_id','time','c_w','c_w_long']]
                df_noisy.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'+noise+'/'+str(trip_id)+'.csv',index=False,header=False)
elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)


