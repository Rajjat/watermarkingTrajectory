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
from random import choices
from scipy.fft import fft, ifft
from math import radians, cos, sin, asin, sqrt
from random import choices

def double_emded(trace):
    lat_array = trace[:,0:2]
    c = []
    for i in range(256):
        c.append(complex(lat_array[i][1], lat_array[i][0])) # first lat, then long

    watermark=np.zeros(256)
    
    colors = [-1,1]
    for i in range(1,200):
        choice=choices(colors, k=1)
        watermark[i]=choice[0]

    x1 = fft(c)
    magnitude = x1.real
    p=float(config['global']['d'])
    new_magnitude=np.zeros(256)
    for j in range(256):
        new_magnitude[j] = magnitude[j] + (p * watermark[j])

    formed_signal = []
    import cmath
    for k in range(256):
        z = complex(new_magnitude[k], x1[k].imag)
        formed_signal.append(z)
    watermarked_signal = np.fft.ifft(formed_signal);
    df_watermarked = pd.DataFrame(watermarked_signal.real, columns=['c_w'])
    df_watermarked['c_w_long'] = watermarked_signal.imag
    return df_watermarked

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcuate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
    
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
        '/data/watermarkingTraj/data/All_results/'+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/watermark_corrWithDistance.csv',header=None)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values
    for trip_id in trip_idSeries:
        data=pd.read_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/watermarking/watermarkedTraj.csv')
        data_three = data[['watermarked_long', 'watermarked_lat','cum_sum','latitude','longitude']]

        noises=['double_emded']
        for noise in noises:
            
            noisy_trace = double_emded(data_three.values)

            traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id
            traj_folder = traj_folder + "/"

            final_folder=traj_folder
            final_folder = final_folder + "/"
            path1=final_folder + "/noise_added"
            if not os.path.isdir(path1):
                os.mkdir(path1)

            compare_dist=data_three[['watermarked_lat','watermarked_long','latitude','longitude','cum_sum']]
            frames=[noisy_trace,compare_dist]
            final_df=pd.concat(frames, axis=1)
            final_df['dist']=final_df.apply(lambda row: haversine(row['watermarked_long'],row['watermarked_lat'],
                                                                  row['c_w_long'],row['c_w']), axis=1)
            final_df['trip_id']=trip_id
 
            final_df.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/noise_added/'+noise+'.csv',index=False)
       
    elapsed_time_fl = (time.time() - start)
    print('elapsed_time_fl=',elapsed_time_fl)
