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
from scipy.linalg import svd
import math
import re
from math import radians, cos, sin, asin, sqrt

def double_embed(df_2):
    delta=0.000001
    new_SV = []
    added_watermark_bit=[]
    watermark = np.load('/data/watermarkingTraj/data/All_results/SVD/watermark_double_embed.npy', allow_pickle=True)
    s_w = df_2[["latitude","longitude"]].values
    for i in range(0, 128):  # P=16, n=32
        lat2d = s_w[2 * i:(2 * i) + 2].copy() #for 256

        u, s, vh = svd(lat2d)

        s_temp=np.zeros(len(s))
        for k in range(0, len(s)):
            s_temp[k] = s[k] ** 2
        temp = 0.0
        for l in range(0, len(s)):
            temp = s_temp[l] + temp
        N_j = math.sqrt(temp)
        Y_j = N_j % delta
        if watermark[i] == 1.0:
            if Y_j < delta / 4:
                N_j_dash = N_j - (delta / 4) - Y_j
                added_watermark_bit.append(1)
            else:
                N_j_dash = N_j + (3 * delta / 4) - Y_j
                added_watermark_bit.append(1)
        elif watermark[i] == 0.0:
            if Y_j < (3 * delta) / 4:
                N_j_dash = N_j + (delta / 4) - Y_j
                added_watermark_bit.append(0)
            else:
                N_j_dash = N_j + (5 * delta / 4) - Y_j
                added_watermark_bit.append(0)
        temp_s = np.zeros(len(s))
        for m in range(0, len(s)):
            temp_s[m] = s[m] * (N_j_dash / N_j)

        
        Sigma_sw = zeros((lat2d.shape[0], lat2d.shape[1]))

        Sigma_sw[:lat2d.shape[1], :lat2d.shape[1]] = diag(temp_s)
        x=u.dot(Sigma_sw.dot(vh))
        new_SV.append(x)
    new_SV_flat = np.array(new_SV).flatten()
    c_w=[]
    c_w_long=[]
    for i in range(len(new_SV_flat)):
        if i%2==0:
            c_w.append(new_SV_flat[i])
        else:
            c_w_long.append(new_SV_flat[i])
    d = {'c_w':c_w,'c_w_long':c_w_long}
    df_watermarked=pd.DataFrame(d)
    df_wc = pd.concat([df_2, df_watermarked], axis=1)

    return df_wc

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
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


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367000 * c
    return m

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
    noises=['double_embed']
    directory = os.path.join("/data/watermarkingTraj/data/All_results/SVD/our_data/watermarked_data/")
    for root,dirs,files in os.walk(directory):
        for file in files:
            df=pd.read_csv("/data/watermarkingTraj/data/All_results/SVD/our_data/watermarked_data/"+file,header=0)
            trip_id=df.traj_id.values[0]
            for noise in noises:
                noisy_trace = locals()[noise](df)
                df_noisy = pd.DataFrame(noisy_trace)
 
                traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'
                traj_folder = traj_folder + "/"
                final_folder=traj_folder
                final_folder = final_folder + "/"
                path1=final_folder + noise
                if not os.path.isdir(path1):
                    os.mkdir(path1)
                df_noisy['dist']=df_noisy.apply(lambda row: haversine_np(row['watermarked_long'],row['watermarked_lat'],
                                                                      row['c_w_long'],row['c_w']), axis=1)
                df_noisy.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'+noise+'/'+str(trip_id)+'.csv',index=False,header=True)
elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)
