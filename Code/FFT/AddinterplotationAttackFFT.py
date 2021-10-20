import random
import numpy as np
from datetime import datetime
import pyforest
import numpy as np
import pandas as pd
import time
from PyEMD import EMD
import numpy  as np
import pandas as pd
import pylab as plt
import pylab as plt1
from numpy import diag
from numpy import zeros
import os
import logging
import argparse
import configparser
import numpy as np
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

       
        a_time = (df.iloc[i:i+1].capture_time.values[0])
        b_time = df.iloc[i+1:i+2].capture_time.values[0]
        td1=(b_time - a_time)
        if int(td1) > 1.5e+10 or int(td1) < 0 :
            print(td1)
            continue;
        noised_lat=(b_lat - a_lat) * t + a_lat
        noised_long=(b_long - a_long) * t +  a_long
        temporal =(b_time - a_time) * t + a_time
        
        line = pd.DataFrame({"trip_id": df.trip_id.values[0], 'longitude':0.0, 'latitude':0.0, 'capture_time':temporal, 'Distance':0,'time_diff':0, 'cum_sum':0, 'watermarked_lat':0.0,'watermarked_long':0.0,'dist':0,'c_w':noised_lat,'c_w_long':
                            noised_long}, index=[i])
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

    global config
    config = configparser.ConfigParser()
    config.read(args.configfile)


    start = time.time()
    df1 = pd.read_csv(
        '/data/watermarkingTraj/data/All_results/'+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/watermark_corrWithDistance.csv',header=None)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values
    for trip_id in trip_idSeries:
        
        data=pd.read_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/watermarking/watermarkedTraj.csv')
        pd.options.mode.chained_assignment = None
        data["capture_time"]=data['capture_time'].astype('datetime64[s]')
        final_df=interpolate(data)
 
        traj_folder = traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id
        traj_folder = traj_folder + "/"

        final_folder=traj_folder
        final_folder = final_folder + "/"
        path1=final_folder + "/noise_added"
        if not os.path.isdir(path1):
            os.mkdir(path1)



        final_df.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/noise_added/interploatationNoise.csv',index=False)

elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)

