import random
from datetime import datetime
import time
import numpy  as np
import pandas as pd
from numpy import diag
from numpy import zeros
import os
import logging
import argparse
import configparser
import numpy as np
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
    df1 = pd.read_csv(
        '/data/watermarkingTraj/data/All_results/'+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/watermark_corrWithDistance.csv',header=None)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values
    for trip_id in trip_idSeries:
        
        data=pd.read_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/watermarking/watermarkedTraj.csv')

        pd.options.mode.chained_assignment = None
        data["capture_time"]=data['capture_time'].astype('datetime64[s]')
        
        final_df=cropfromlast(data)

        traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id
        traj_folder = traj_folder + "/"

        final_folder=traj_folder
        final_folder = final_folder + "/"
        path1=final_folder + "/noise_added"
        if not os.path.isdir(path1):
            os.mkdir(path1)



        final_df.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'//payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/noise_added/cropfromlast.csv',index=False)

elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)

