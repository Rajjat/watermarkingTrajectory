
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
#from scipy import stats

#delta = 0.0009


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

def getwatermark2():
    watermark = np.zeros(256)
    for i in range(1, 999):
        if i % 2 == 0:
            watermark[i] = 1
        else:
            watermark[i] = -1
    print('watermark sum=',sum(watermark))
    return watermark

def getwatermark3():
    watermark = np.zeros(256)
    for i in range(1, 99):
        if i % 2 == 0:
            watermark[i] = 1
        else:
            watermark[i] = -1
    print('watermark sum=',sum(watermark))
    return watermark

def getwatermark():

    watermark=np.zeros(256)
    
    colors = [-1,1]
    for i in range(256):
        choice=choices(colors, k=1)
            #print(choice[0])
        watermark[i]=choice[0]
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
    lp=int(config['global']['slice'])
    lat_array = df_2[['latitude', 'longitude']].values
    c = []
    for i in range(lp):
        c.append(complex(lat_array[i][0], lat_array[i][1]))

    x1 = fft(c)
    magnitude = x1.real
    p=float(config['global']['d'])
    new_magnitude=np.zeros(lp)
    for j in range(lp):
        new_magnitude[j] = magnitude[j] + (p * watermark[j])
    formed_signal = []
    import cmath
    for k in range(lp):
        z = complex(new_magnitude[k], x1[k].imag)
        formed_signal.append(z)
    watermarked_signal = np.fft.ifft(formed_signal);
    df_watermarked = pd.DataFrame(watermarked_signal.real, columns=['watermarked_lat'])
    df_watermarked['watermarked_long'] = watermarked_signal.imag
    frame = [df_2, df_watermarked]
    final_df = pd.concat(frame, axis=1)
    return final_df,x1


def watermarkExtract(df_wc,x1_org,watermark,trip_id):
    lp=int(config['global']['slice'])
    watermarked_array = df_wc[['watermarked_lat', 'watermarked_long']].values
    c_w = []
    for l in range(lp):
        c_w.append(complex(watermarked_array[l][0], watermarked_array[l][1]))

    p=float(config['global']['d'])
    fourier_wc = fft(c_w)
    extract_wc = np.zeros(lp)
    for m in range(lp):
        vl = (((fourier_wc[m].real - x1_org[m].real)) / p)
        extract_wc[m]=vl
    return extract_wc

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
    filterdata = df.groupby("trip_id").filter(lambda x: len(x) > 256)
    filterdata = filterdata.reset_index(drop=True)
    trip_idSeries = filterdata['trip_id'].unique()
    #watermark = getwatermark()
    #np.save("/data/watermarkingTraj/data/All_results/FFT_complexNum/watermark_random.npy",watermark)
    lp=int(config['global']['slice'])     i=0
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
        appended_data = []
        watermark_256=[]
        extracted_watermark_256=[]
        if max1<=200:
            for i in range(16):
                watermark=np.load("/data/watermarkingTraj/data/All_results/FFT_complexNum/256_len/"+config['global']['slicenumber']+"/payload_strength/different_watermarks/payload_5/watermark_5.npy",allow_pickle=True)                
                watermark=watermark[0:16]
                watermark_256.append(watermark)
                df_3=df_2[lp*i:lp*i+lp]
                df_3=df_3.reset_index(drop=True)
                df_wc,x1  = addwatermark(watermark,df_3,trip_id)
                
                df_wc['dist'] = df_wc.apply(lambda row: haversine(row['watermarked_long'], row['watermarked_lat'],
                                                            row['longitude'], row['latitude']), axis=1)
                appended_data.append(df_wc)
                extract_wc=watermarkExtract(df_wc,x1,watermark,trip_id)
                extracted_watermark_256.append(extract_wc)
                corr_watermark = ncc(np.array(extract_wc).flatten(), np.array(watermark).flatten())
                print('inside',corr_watermark)
                

            appended_data = pd.concat(appended_data)
            appended_data=appended_data.reset_index(drop=True)
            traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'
            traj_folder = traj_folder + "/"
            path = traj_folder+trip_id
            if not os.path.isdir(path):
                os.mkdir(path)
            final_folder = path + "/"
            path1 = final_folder + "watermarking"
            if not os.path.isdir(path1):
                os.mkdir(path1)
            appended_data.to_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/watermarking/watermarkedTraj.csv',index=False)
            corr_watermark = ncc(np.array(watermark_256), np.array(extracted_watermark_256))
            avg = appended_data["dist"].mean()
            max1 = appended_data["dist"].max()
            min1 = appended_data["dist"].min()
            path='../'+config['global']['global_fol']+config['global']['technique']
            with open(path+'//256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermark_corrWithDistance.csv', 'a') as fd:
                myCsvRow=trip_id + ',' + "{0:0.4f}".format(avg) + ',' + "{0:0.4f}".format(min1) + ',' + "{0:0.4f}".format(max1)+ ',' + "{0:0.4f}".format(corr_watermark)+'\n'
                fd.write(myCsvRow)
        else:
            continue

    elapsed_time_fl = (time.time() - start)
    print('elapsed_time_fl=',elapsed_time_fl)


