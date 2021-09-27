
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

def addwatermark(df_2):
    lat_array = df_2[['latitude', 'longitude']].values
    #print(len(lat_array))
    c = []
    lp=int(config['global']['slice']) 
    for i in range(lp):
        c.append(complex(lat_array[i][0], lat_array[i][1]))

    x1 = fft(c)
    return x1

def watermarkExtract(watermark,trip_id):
    df_concat=pd.DataFrame(columns=['traj_id','Noise','corr_value','avg','min','max'])
    df_noise = pd.DataFrame(columns=['traj_id','Noise','corr_value','avg','min','max'])
    noises=['double_emded']
    for noise in noises:
        df_wc = pd.read_csv(
            '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/noise_added/'+noise+'.csv')
        x1 = addwatermark(df_wc)
        if trip_id==df_wc['trip_id'][0]:
            watermarked_array = df_wc[['c_w', 'c_w_long']].values
            c_w = []
            for l in range(256):
                c_w.append(complex(watermarked_array[l][0], watermarked_array[l][1]))
            #print('c_w=', c_w)
            p = float(config['global']['d'])
            fourier_wc = fft(c_w)
            extract_wc = []
            for m in range(256):
                vl = (((fourier_wc[m].real - x1[m].real)) / p)
                extract_wc.append(vl)


            corr_watermark = np.corrcoef(np.array(extract_wc).flatten(), watermark)
            df_wc['dist'] = df_wc.apply(lambda row: haversine(row['c_w_long'], row['c_w'],
                                                                row['watermarked_long'], row['watermarked_lat']), axis=1)

            avg = df_wc["dist"].mean()
            max1 = df_wc["dist"].max()
            min1 = df_wc["dist"].min()
            df_noise = df_noise.append({'traj_id': trip_id,'Noise':noise,'corr_value':"{0:0.4f}".format(corr_watermark[0][1]),
                                        'avg':"{0:0.4f}".format(avg),'max':"{0:0.4f}".format(max1),'min':"{0:0.4f}".format(min1)}, ignore_index=True)
            df_noise.to_csv(
            '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+trip_id+'/watermarking/watermark_cor_withNoises_'+noise+'.csv',
            index=False)
    return df_noise

def watermarkExtract1(watermark,x1,data_wc):
    lp=int(config['global']['slice']) 
    #noise='cropfromlast'
    p = float(config['global']['d'])
    watermarked_array2 = data_wc[['c_w', 'c_w_long']].values
    c_w2=[]
    for f in range(lp):
        c_w2.append(complex((watermarked_array2[f][0]), (watermarked_array2[f][1])))
    fourier_wc2 = fft(c_w2) #watermarked

    extract_wc2=np.zeros(lp)
    for m in range(lp):
        vl2 = ((((fourier_wc2[m].real) - (x1[m].real)) / p))
        extract_wc2[m]=vl2

    return extract_wc2

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
    start = time.time()

    watermark = np.load('/data/dadwal/watermarkingTraj/data/All_results/SVD/watermark.npy', allow_pickle=True)
    watermark=watermark.flatten()
    d = float(config['global']['d'])

    noises=['remove_random_points_with_path','add_outliers_with_signal_to_noise_ratio','add_signal_noise',
        'add_white_noise','replace_non_skeleton_points_with_path','replace_non_skeleton_points_with_start_point',
        'remove_random_points','hybrid','cropfromlast','double_embed','interpolate']
    for noise in noises:
        directory = os.path.join("/data/dadwal/watermarkingTraj/data/All_results/SVD/our_data/noise_traj/"+noise+"/")
        df_concat1=pd.DataFrame(columns=['traj_id','Noise','corr_value'])
        for root,dirs,files in os.walk(directory):
            for file1 in files:                
                df=pd.read_csv("/data/dadwal/watermarkingTraj/data/All_results/SVD/our_data/noise_traj/"+noise+"/"+file1,header=0)
                df_noise=pd.DataFrame(columns=['traj_id','Noise','corr_value'])
                file_split=re.split("\.",file1)
                trip_id=file_split[0]
                extracted_watermark=watermarkExtract(df,d,watermark)
                corr_watermark2=ncc(watermark.flatten(), np.array(extracted_watermark).flatten())
                df_noise = df_noise.append({'traj_id': trip_id,'Noise':noise,'corr_value':"{0:0.4f}".format(corr_watermark2)}, ignore_index=True)
                df_concat1 = pd.concat([df_concat1, df_noise])

        df_concat1.to_csv('../'+config['global']['global_fol'] + config['global']['technique'] + '/our_data/watermark_extract/watermark_corr_'+noise+'.csv',index=False)