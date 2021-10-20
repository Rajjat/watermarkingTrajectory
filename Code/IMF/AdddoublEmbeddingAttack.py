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

def double_emded(df_2):
    #lat_array = trace[:,0:2]
    #print(lat_array)


    watermark=np.empty(shape=(16,16))
    
    colors = [-1,1]
    for i in range(1,16):
        for j in range(1,16):
            choice=choices(colors, k=1)
            #print(choice[0])
            watermark[i][j]=choice[0]
    

    watermark_on=config['global']['watermark_on']
    t = df_2.cum_sum.values
    s = df_2.watermarked_latitude.values

    IMF_lat = EMD().emd(s, t)
    selectIMF = int(config['global']['select_IMF'])
    # 2. Choose one IMF amd convert it  from 1D to 2D (C matrix).
    # IMF_lat_1 = np.reshape(IMF_lat[0], (-1, 32))
    #global matrix_size
    matrix_size=int(config['global']['matrix_size'])
    IMF_lat_1 = np.reshape(IMF_lat[selectIMF], (-1, matrix_size))
    #print(IMF_lat_1.shape)
    # 3. The 2D IMF matrix C is decomposed by singular value decomposition
    u, s, vh = np.linalg.svd(IMF_lat_1, full_matrices=False)
   
    watermark = watermark[0:matrix_size, 0:matrix_size]
    d = float(config['global']['d'])
   
    dW = watermark * d
   
    Sigma = zeros((matrix_size, matrix_size))
    Sigma[:IMF_lat_1.shape[1], :IMF_lat_1.shape[1]] = diag(s)
    
    D = Sigma + dW
    
    # 5 The SVD is applied in the matrix D.
    u_w, s_w, vh_w = np.linalg.svd(D)

    Sigma_sw = zeros((matrix_size, matrix_size))
    Sigma_sw[:IMF_lat_1.shape[1], :IMF_lat_1.shape[1]] = diag(s_w)

    c_w = u.dot(Sigma_sw.dot(vh))

    # 7 add watermark IMF with other IMFs
    c_w_flatten = c_w.flatten()
    no_of_zeros=len(c_w_flatten)
    # IMF_watermarked = c_w_flatten + IMF_lat[1] + IMF_lat[2] + IMF_lat[3] + IMF_lat[4] + IMF_lat[5]
    # IMF_watermarked = c_w_flatten+  IMF_lat[1] +IMF_lat[2]#+ IMF_lat[3] + IMF_lat[4]+IMF_lat[5]
    IMF_watermarked = np.zeros(no_of_zeros)
    for i in range(len(IMF_lat)):
        if i == selectIMF:
            IMF_watermarked += c_w_flatten
        else:
            IMF_watermarked += IMF_lat[i]
    IMF_watermarked = IMF_watermarked.flatten()
    #IMF_watermarked.to_csv()
    
    #df_watermarked = pd.DataFrame(IMF_watermarked, columns=[watermarked_cor])    
        
        
    df_watermarked = pd.DataFrame(IMF_watermarked.real, columns=['c_w'])
    df_watermarked['c_w_long'] = df_2['longitude'].copy()
    return df_watermarked

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
        data_three = data[['longitude', 'watermarked_latitude','cum_sum','latitude']]

        noises=['double_emded']
        for noise in noises:
            #noisy_trace = locals()[config['global']['select_noise']](data_three.values)
            
            noisy_trace = double_emded(data_three)

            traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id
            traj_folder = traj_folder + "/"

            final_folder=traj_folder
            final_folder = final_folder + "/"
            path1=final_folder + "/noise_added"
            if not os.path.isdir(path1):
                os.mkdir(path1)

            compare_dist=data_three[['watermarked_latitude','latitude','longitude','cum_sum']]
            frames=[noisy_trace,compare_dist]
            final_df=pd.concat(frames, axis=1)
            final_df['dist']=final_df.apply(lambda row: haversine(row['longitude'],row['watermarked_latitude'],
                                                                  row['c_w_long'],row['c_w']), axis=1)
            final_df['trip_id']=trip_id
 
            final_df.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/noise_added/'+noise+'.csv',index=False)
            #print(final_df)

            #t = data_three.cum_sum.values
            #s = data_three.watermarked_lat.values
       
    elapsed_time_fl = (time.time() - start)
    print('elapsed_time_fl=',elapsed_time_fl)
