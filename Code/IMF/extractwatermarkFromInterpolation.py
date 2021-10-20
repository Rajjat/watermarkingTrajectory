from PyEMD import EMD
import pandas as pd
import pylab as plt
from numpy import diag
from numpy import zeros
import os
import argparse
import configparser
import numpy as np
import time
import statistics
import math
import seaborn as sns
from scipy.fft import fft, ifft
from math import radians, cos, sin, asin, sqrt


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
def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


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

def addwatermark(df_2,watermark):
    df_3=df_2.loc[df_2['latitude']!=0.0]

    watermark_on=config['global']['watermark_on']
    t = df_3.cum_sum.values
    s = df_3[watermark_on].values

    IMF_lat = EMD().emd(s, t)
    selectIMF = int(config['global']['select_IMF'])
    matrix_size=int(config['global']['matrix_size'])
    IMF_lat_1 = np.reshape(IMF_lat[selectIMF], (-1, matrix_size))
    u, s, vh = np.linalg.svd(IMF_lat_1, full_matrices=False)
    watermark = np.reshape(watermark, (-1, matrix_size))
    d = float(config['global']['d'])
   
    dW = watermark * d
   
    Sigma = zeros((matrix_size, matrix_size))
    Sigma[:IMF_lat_1.shape[1], :IMF_lat_1.shape[1]] = diag(s)
    
    D = Sigma + dW
    
    # 5 The SVD is applied in the matrix D.
    u_w, s_w, vh_w = np.linalg.svd(D)

    return u_w, vh_w, Sigma


def watermarkExtract(watermark,trip_id):
    df_noise = pd.DataFrame(columns=['traj_id','Noise','corr_value'])
    noises=['interploatationNoise']
    #p = float(config['global']['d'])
    matrix_size = int(config['global']['matrix_size'])
    selectIMF = int(config['global']['select_IMF'])
    d = float(config['global']['d'])
 
    for noise in noises:
        df_wc = pd.read_csv(
            '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/noise_added/'+noise+'.csv')
        if trip_id==df_wc['trip_id'][0]:
            ls=[]
            j=0
            index=0
            watermarked_lat =df_wc.watermarked_latitude.values
            watermarked_lat1= watermarked_lat[watermarked_lat!= 0]
           

            watermarked_long = df_wc.longitude.values
            watermarked_long1= watermarked_long[watermarked_long!= 0]
            
            c_w = df_wc.c_w.values
            c_w_long= df_wc.c_w_long.values
            #print(c_w)
            time_cum=df_wc.cum_sum.values
            for i in range(256):
                dist=[]
                for j in range(len(df_wc.index)):
                    dist.append(haversine_np(watermarked_long1[i],watermarked_lat1[i],c_w_long[j],c_w[j]))

                index_min = np.argmin(dist)
                temp=(c_w_long[index_min],c_w[index_min],time_cum[index_min])
                ls.append(temp)


            u_w, vh_w, Sigma = addwatermark(df_wc,watermark)
            t=np.zeros(256)
            s=np.zeros(256)
            for i in range(256):
                t[i]=ls[i][2]
                s[i]=ls[i][1]
               
            # 1. Execute EMD on signal
            IMF_latstar = EMD().emd(s, t)
            #global selectIMF
            IMF_lat_star = np.expand_dims(IMF_latstar[selectIMF], axis=0)

            # 2. The obtained 1D IMF component is transformed into a 2D matrix
            c_star_w = np.reshape(IMF_lat_star, (-1, matrix_size))
            # 3. The SVD is performed on the IMF component C_star_W .
            u_star, s_star, vh_star = np.linalg.svd(c_star_w, full_matrices=False)
            # 4 The matrix, which contains the watermark, is evaluated
            Sigma_star = zeros((c_star_w.shape[0], c_star_w.shape[1]))
            Sigma_star[:c_star_w.shape[1], :c_star_w.shape[1]] = diag(s_star)

            d_star = u_w.dot(Sigma_star.dot(vh_w))
            # 5. The corrupted watermark is obtained.
            d_starSigma = np.subtract(d_star, Sigma)

            W_star = d_starSigma / d
            corr_watermark = ncc(W_star.flatten(), watermark.flatten())
            df_noise = df_noise.append({'traj_id': trip_id,'Noise':noise,'corr_value':"{0:0.4f}".format(corr_watermark)}
        , ignore_index=True)
            df_noise.to_csv(
            '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/watermarking/watermark_cor_withNoises_'+noise+'.csv',
            index=False)
        return df_noise




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

    start = time.time()

    df1 = pd.read_csv('/data/watermarkingTraj/data/All_results/IMF/256_len/watermark_corrWithDistance_finalused.csv',header=None)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values

    i=0

    watermark=np.load("/data/watermarkingTraj/data/All_results/FFT_complexNum//256_len/watermark_random_sumZero_256.npy",allow_pickle=True)
    df_concat1 = pd.DataFrame(columns=['traj_id', 'Noise', 'corr_value'])
    
    noises='interploatationNoise'
    for trip_id in trip_idSeries:
        df=watermarkExtract(watermark,trip_id)
        df_concat1 = pd.concat([df_concat1, df])

        df_concat1.to_csv('../'+config['global']['global_fol'] + config['global']['technique'] + '/256_len/watermark_cor_withNoises_'+noises+'.csv',
            index=False)



