import random
from datetime import datetime
import numpy as np
import pandas as pd
import time
from PyEMD import EMD
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
    #df_noise = pd.DataFrame(columns=['traj_id','Noise','watermarked_latitude','longitude','capture_time'])
    #locs=[]

    np.random.seed(0)
    #fmt = '%Y-%m-%d %H:%M:%S'
    index1=random.sample(range(2,255), 3)
    
    df2=df.copy()
    #for row, index in df.iteritems():
     #   df2.loc[row,'c_w']=df.watermarked_lat.values[0]
        
    df2['c_w']=df['watermarked_latitude']
    df2['c_w_long']=df['longitude']
    for i in index1:
        #print(i)
        t=random.uniform(0.0, 0.0002)
        #i=index1[k]
        #t = slots[j]
        a_lat = float(df.iloc[i:i+1].watermarked_latitude.values[0])
        b_lat = float(df.iloc[i+1:i+2].watermarked_latitude.values[0])

        a_long = df.iloc[i:i+1].longitude.values[0]
        b_long = df.iloc[i+1:i+2].longitude.values[0]

       
        a_time = (df.iloc[i:i+1].capture_time.values[0])
        #print(type(a_time))
        b_time = df.iloc[i+1:i+2].capture_time.values[0]
        #print('bad m')
        #print(type(a_time))
        #a_time= datetime.fromisoformat(df.iloc[i:i+1].capture_time.values[0])
        #b_time = datetime.fromisoformat(df.iloc[i+1:i+2].capture_time.values[0])
        #print((b_time))
        #print(a_lat)
        td1=(b_time - a_time)
       
        #td1=int(round(td.total_seconds()))
        if int(td1) > 1.5e+10 or int(td1) < 0 :
            print(td1)
            continue;
        noised_lat=(b_lat - a_lat) * t + a_lat
        noised_long=(b_long - a_long) * t +  a_long
        temporal =(b_time - a_time) * t + a_time
        

        #c =( (b_time - a_time) * t + a_time , (b_lat - a_lat) * t + a_lat, (b_long - a_long) * t + a_long)
        #locs.append(c);
        line = pd.DataFrame({"trip_id": df.trip_id.values[0], 'longitude':0.0, 'latitude':0.0, 'capture_time':temporal, 'Distance':0,'time_diff':0, 'cum_sum':0, 'watermarked_latitude':0.0,'dist':0,'c_w':noised_lat,'c_w_long':noised_long}, index=[i])
        df2 = pd.concat([df2.iloc[:i], line, df2.iloc[i:]]).reset_index(drop=True)
    return df2 



if __name__ == "__main__":
    #Traj_000041f9e823d8a30a70b408c04f894e
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
        '/data/watermarkingTraj/data/All_results/'+config['global']['technique']+'/256_len/watermark_corrWithDistance_finalused.csv',header=None)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values
    for trip_id in trip_idSeries:
        
        data=pd.read_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/watermarking/watermarkedTraj.csv')
    #    print('Before noise')
        #print(data.head())
        #data_three = data[['longitude', 'watermarked_latitude','cum_sum','latitude']]
        pd.options.mode.chained_assignment = None
        data["capture_time"]=data['capture_time'].astype('datetime64[ns]')
        final_df=interpolate(data)
 


        traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id
        traj_folder = traj_folder + "/"

        final_folder=traj_folder
        final_folder = final_folder + "/"
        path1=final_folder + "/noise_added"
        if not os.path.isdir(path1):
            os.mkdir(path1)



        final_df.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/noise_added/interploatationNoise.csv',index=False)

        #t = data_three.cum_sum.values
        #s = data_three.watermarked_lat.values
elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)

