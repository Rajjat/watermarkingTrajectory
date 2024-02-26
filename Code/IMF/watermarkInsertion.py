
import pandas as pd
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
from random import choices
from PyEMD import EMD
from numpy import diag
from numpy import zeros
import os



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


def getwatermark1():
    watermark = np.zeros(256)
    for i in range(1, 899):
        if i % 2 == 0:
            watermark[i] = 1
        else:
            watermark[i] = -1
    #print('watermark sum=',(watermark))
    return watermark

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

def distance(x):
    y = x.shift()
    return haversine_np(x['longitude'], x['latitude'], y['longitude'], y['latitude']).fillna(0)


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.H * m)


def addwatermark(watermark,df_2,trip_id):
    # 1. Convert signal into IMF
    watermark_on=config['global']['watermark_on']
    t = df_2.cum_sum.values
    s = df_2[watermark_on].values

    IMF_lat = EMD().emd(s, t)
    selectIMF = int(config['global']['select_IMF'])
    # 2. Choose one IMF amd convert it  from 1D to 2D (C matrix).
    matrix_size=int(config['global']['matrix_size'])
    IMF_lat_1 = np.reshape(IMF_lat[selectIMF], (-1, matrix_size))
    #print(IMF_lat_1.shape)
    # 3. The 2D IMF matrix C is decomposed by singular value decomposition
    u, s, vh = np.linalg.svd(IMF_lat_1, full_matrices=False)

    # 4. The watermark (W matrix) is embedded to the SVs of the IMF matrix.
    watermark = watermark[0:matrix_size, 0:matrix_size]
    d = float(config['global']['d'])
   
    dW = watermark * d
   
    Sigma = zeros((matrix_size, matrix_size))
    Sigma[:IMF_lat_1.shape[1], :IMF_lat_1.shape[1]] = diag(s)
    
    D = Sigma + dW
    
    # 5 The SVD is applied in the matrix D.
    u_w, s_w, vh_w = np.linalg.svd(D)
   # 6 The watermarked IMF component (C W matrix) in 2D is acquired by using a modified matrix of SVs (s_w matrix).
    # Sigma_sw = zeros((32, 32))
    Sigma_sw = zeros((matrix_size, matrix_size))
    Sigma_sw[:IMF_lat_1.shape[1], :IMF_lat_1.shape[1]] = diag(s_w)

    c_w = u.dot(Sigma_sw.dot(vh))

    # 7 add watermark IMF with other IMFs
    c_w_flatten = c_w.flatten()
    no_of_zeros=len(c_w_flatten)
    IMF_watermarked = np.zeros(no_of_zeros)
    for i in range(len(IMF_lat)):
        if i == selectIMF:
            IMF_watermarked += c_w_flatten
        else:
            IMF_watermarked += IMF_lat[i]
    IMF_watermarked = IMF_watermarked.flatten()
    #IMF_watermarked.to_csv()
    watermarked_cor='watermarked_'+config['global']['watermark_on']
    df_watermarked = pd.DataFrame(IMF_watermarked, columns=[watermarked_cor])
    
    
    
    append_watermarked = pd.concat([df_2, df_watermarked], axis=1)
    
    append_watermarked['dist'] = append_watermarked.apply(lambda row: haversine(row['longitude'],
                                                                                    row[watermarked_cor],
                                                            row['longitude'], row['latitude']), axis=1)


    traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'
    traj_folder = traj_folder + "/"
    path = traj_folder+trip_id
    if not os.path.isdir(path):
        os.mkdir(path)
    final_folder = path + "/"
    path1 = final_folder + "watermarking"
    if not os.path.isdir(path1):
        os.mkdir(path1)
    append_watermarked.to_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/watermarking/watermarkedTraj.csv',
        index=False)
    return append_watermarked,u_w, vh_w, Sigma


def watermarkExtract(df_watermarked,u_w, vh_w, Sigma,watermark,trip_id):
    matrix_size = int(config['global']['matrix_size'])
    selectIMF = int(config['global']['select_IMF'])
    d = float(config['global']['d'])
    watermarked_cor='watermarked_'+config['global']['watermark_on']
    t = df_watermarked.cum_sum.values
    s = df_watermarked.watermarked_latitude.values
    ############
    # 1. Execute EMD on signal
    IMF_latstar = EMD().emd(s, t)
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
   
    # 6 compare extracted watermark with original watermark
    corr_watermark = ncc(watermark.flatten(), W_star.flatten())
    avg = df_watermarked["dist"].mean()
    max1 = df_watermarked["dist"].max()
    min1 = df_watermarked["dist"].min()
    #print('watermark cor with watermark traj only=', corr_watermark)
    path='../'+config['global']['global_fol']+config['global']['technique']
    with open(path+'/256_len/watermark_corrWithDistance.csv', 'a') as fd:
        myCsvRow=trip_id + ',' + "{0:0.4f}".format(avg) + ',' + "{0:0.4f}".format(min1) + ',' + "{0:0.4f}".format(max1)+ ',' + "{0:0.4f}".format(corr_watermark)+'\n'
        fd.write(myCsvRow)
        
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
    df = pd.read_csv(
        '../'+config['global']['global_folder']  + 'data_watermarking.csv')
    df = pd.read_csv('../'+config['global']['global_folder']  + 'data_watermarking.csv')
    filterdata = df.groupby("trip_id").filter(lambda x: len(x) > 256)
    filterdata = filterdata.reset_index(drop=True)
    trip_idSeries = filterdata['trip_id'].unique()
    watermark=np.load("/data/watermarkingTraj/data/All_results/IMF/256_len//watermark_random_sumZero_256.npy",allow_pickle=True)
    matrix_size = int(config['global']['matrix_size'])
    watermark1 = np.reshape(watermark, (-1, matrix_size))
    i=0
    for trip_id in trip_idSeries:
        df_1=filterdata.loc[filterdata['trip_id']==trip_id]
        df_1 = df_1.reset_index(drop=True)
        df_2=df_1.head(256)
        df_2['Distance'] = distance(df_2).reset_index(level=0, drop=True)
        max1=df_2['Distance'].max()
        
        pd.options.mode.chained_assignment = None
        df_2["capture_time"]=df_2['capture_time'].astype('datetime64[ns]')
        df_2['time_diff'] = df_2["capture_time"].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
        df_2['cum_sum'] = df_2['time_diff'].cumsum()
                
        if max1<=200:                   
            final_df,u_w, vh_w, Sigma  = addwatermark(watermark1,df_2,trip_id)
            watermarkExtract( final_df,u_w, vh_w, Sigma,watermark1,trip_id)
        else:
            continue
    elapsed_time_fl = (time.time() - start)
    print('elapsed_time_fl=',elapsed_time_fl)


