
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
import re
from scipy.linalg import svd
#from scipy import stats
#delta = 0.0009


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

def watermarkExtract(df_wc,delta,watermark):
    s_w = df_wc[["c_w","c_w_long"]].values
    wc_found=[]
    for i in range(0, 128):  # P=16, n=32
        lat2d = s_w[2 * i:(2 * i) + 2].copy()
        u, s, vh =svd(lat2d)
        s_temp=np.zeros(len(s))
        for j in range(0, len(s)):
            s_temp[j] = s[j] ** 2
        temp = 0.0
        for k in range(0, len(s)):
            temp = s_temp[k] + temp
        N_j = math.sqrt(temp)
        #print('N_j=', N_j)
        Y_j = N_j % delta
        if Y_j < (delta / 2):

            wc_found.append(-1)
        else:
            wc_found.append(1)

    return wc_found


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

    watermark = np.load('/data/watermarkingTraj/data/All_results/SVD/watermark.npy', allow_pickle=True)
    #print(watermark)
    watermark=watermark.flatten()
    d = float(config['global']['d'])
    noises=['interpolate']
    for noise in noises:
        directory = os.path.join("/data/watermarkingTraj/data/All_results/SVD/portodata/noise_traj/"+noise+"/")
        df_concat1=pd.DataFrame(columns=['traj_id','Noise','corr_value'])
        for root,dirs,files in os.walk(directory):
            for file1 in files:                
                df_2=pd.read_csv("/data/watermarkingTraj/data/All_results/SVD/portodata/noise_traj/"+noise+"/"+file1,header=0)
                df_noise=pd.DataFrame(columns=['traj_id','Noise','corr_value'])
                file_split=re.split("\.",file1)
                trip_id=file_split[0]
                df_noise = pd.DataFrame(columns=['traj_id','Noise','corr_value'])
                watermark_256=[]
                extracted_watermark_256=[]
                corr_watermark_arr=[]
                ls=[]
                ls1=[]
                ls2=[]
                j=0
                index=0
                watermarked_lat =df_2.watermarked_lat.values
                watermarked_lat1= watermarked_lat[watermarked_lat!= 0]


                watermarked_long = df_2.watermarked_long.values
                watermarked_long1= watermarked_long[watermarked_long!= 0]

                latitude=df_2.latitude.values
                longitude=df_2.longitude.values

                c_w = df_2.c_w.values
                c_w_long= df_2.c_w_long.values
                index_ls=[]
                for i in range(256):
                    dist=[]
                    for j in range(len(df_2.index)):
                        dist.append(haversine_np(watermarked_long1[i],watermarked_lat1[i],c_w_long[j],c_w[j]))

                    index_min = np.argmin(dist)
                    index_ls.append(index_min)
                    min_val=min(dist)
                    temp=(c_w[index_min],c_w_long[index_min])
                    temp1=(latitude[index_min],longitude[index_min])
                    temp3=(watermarked_lat[index_min],watermarked_long[index_min])

                    ls.append(temp)
                    ls1.append(temp1)
                    ls2.append(temp3)

                orginal_traj=pd.DataFrame(ls1, columns=['latitude','longitude'])
                noised_matched=pd.DataFrame(ls, columns=['c_w','c_w_long'])
                watermarked_tr=pd.DataFrame(ls, columns=['watermarked_lat','watermarked_long'])
                t1=pd.concat([orginal_traj, watermarked_tr], axis=1)
                t2=pd.concat([t1, noised_matched], axis=1)
                
                extracted_watermark=watermarkExtract(t2,d,watermark)
                corr_watermark2=ncc(watermark.flatten(), np.array(extracted_watermark).flatten())
                df_noise = df_noise.append({'traj_id': trip_id,'Noise':noise,'corr_value':"{0:0.4f}".format(corr_watermark2)}, ignore_index=True)
                df_concat1 = pd.concat([df_concat1, df_noise])

        df_concat1.to_csv('../'+config['global']['global_fol'] + config['global']['technique'] + '/portodata/watermark_extract/watermark_corr_'+noise+'.csv',index=False)         


