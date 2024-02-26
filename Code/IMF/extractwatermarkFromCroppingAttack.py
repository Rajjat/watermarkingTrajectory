from PyEMD import EMD
from PyEMD import EEMD
import pandas as pd
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
    watermark_on=config['global']['watermark_on']
    t = df_2.cum_sum.values
    s = df_2[watermark_on].values

    IMF_lat = EMD().emd(s, t)
    selectIMF = int(config['global']['select_IMF'])

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

    return u_w, vh_w, Sigma


   
def watermarkExtract1(watermark,trip_id):
    df_noise = pd.DataFrame(columns=['traj_id','Noise','corr_value'])
    noise='cropfromlast'
    p = float(config['global']['d'])
    selectIMF = int(config['global']['select_IMF'])
    d = float(config['global']['d'])
    
    data_wc = pd.read_csv(
        '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/noise_added/'+noise+'.csv')
    data_wc.loc[253:253,'c_w']=data_wc.loc[252:252,'watermarked_latitude'].values
    data_wc.loc[254:254,'c_w']=data_wc.loc[252:252,'watermarked_latitude'].values
    data_wc.loc[255:255,'c_w']=data_wc.loc[252:252,'watermarked_latitude'].values

    data_wc.loc[253:253,'c_w_long']=data_wc.loc[252:252,'longitude'].values
    data_wc.loc[254:254,'c_w_long']=data_wc.loc[252:252,'longitude'].values
    data_wc.loc[255:255,'c_w_long']=data_wc.loc[252:252,'longitude'].values
    u_w, vh_w, Sigma = addwatermark(data_wc,watermark)

    #print(df_watermarked)
    #watermarked_cor='watermarked_'+config['global']['watermark_on']
    t = data_wc.cum_sum.values
    s = data_wc.c_w.values
    ############
    # 1. Execute EMD on signal
    IMF_latstar = EMD().emd(s, t)
    ## get first IMF as we have applied watermark on the first
    # IMF_lat_star = np.expand_dims(IMF_latstar[0], axis=0)
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
    corr_watermark2 = ncc(W_star.flatten(), watermark.flatten())
    df_noise = df_noise.append({'traj_id': trip_id,'Noise':noise,'corr_value':"{0:0.4f}".format(corr_watermark2)}
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
    #print(config['global']['data'])
    start = time.time()

    df1 = pd.read_csv(
        '/data/watermarkingTraj/data/All_results/IMF/256_len/watermark_corrWithDistance_finalused.csv',header=None)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values

    i=0
    watermark=np.load("/data/watermarkingTraj/data/All_results/IMF/256_len//watermark_random_sumZero_256.npy",allow_pickle=True)
    matrix_size = int(config['global']['matrix_size'])
    watermark1 = np.reshape(watermark, (-1, matrix_size))
    df_concat1 = pd.DataFrame(columns=['traj_id', 'Noise', 'corr_value'])
    i=0
    for trip_id in trip_idSeries:
        df=watermarkExtract1(watermark1,trip_id)
        df_concat1 = pd.concat([df_concat1, df])
    df_concat1.to_csv( '../'+config['global']['global_fol'] + config['global']['technique'] + '/256_len/watermark_cor_withNoises'+'.csv',
            index=False)
