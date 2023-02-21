from PyEMD import EMD
from PyEMD import EEMD
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
from scipy.linalg import svd
import math
import re
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
def distance(x):
    y = x.shift()
    return haversine_np(x['longitude'], x['latitude'], y['longitude'], y['latitude']).fillna(0)

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


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.H * m)
def addwatermark(df_2,delta,watermark):
    new_SV = []
    added_watermark_bit=[]
    s_w = df_2[["latitude","longitude"]].values
    selectIMF = int(config['global']['select_IMF'])
    for i in range(0, 128):  # P=16, n=32
        #lat2d = s_w[16 * i:(16 * i) + 16].copy() for 1024
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
        elif watermark[i] == -1.0:
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
    d = {'watermarked_lat':c_w,'watermarked_long':c_w_long}
    df_watermarked=pd.DataFrame(d)
    df_wc = pd.concat([df_2, df_watermarked], axis=1)
    #print('added_watermark_bit=',len(added_watermark_bit))
    return df_wc


def watermarkExtract(df_wc,delta,trip_id,watermark):
    s_w = df_wc[["watermarked_lat","watermarked_long"]].values
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
        Y_j = N_j % delta
        if Y_j < (delta / 2):
            wc_found.append(-1)
        else:
            wc_found.append(1)

    corr_watermark = ncc(watermark.flatten(), np.array(wc_found).flatten())
    path='../'+config['global']['global_fol']+config['global']['technique']
    with open(path+'/our_data/watermark_corrWithDistance.csv', 'a') as fd:
        myCsvRow=trip_id + ',' + "{0:0.4f}".format(corr_watermark)+'\n'
        fd.write(myCsvRow)
    print("corr_watermark=",corr_watermark)
    #print(coor_watermark1)


        

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

    directory = os.path.join("/data/watermarkingTraj/data/All_results/TrajGuard/our_data/data/")
    watermark = np.load('/data/watermarkingTraj/data/All_results/SVD/watermark.npy', allow_pickle=True)
    watermark=watermark.flatten()
    d = float(config['global']['d'])
    i=0
    for root,dirs,files in os.walk(directory):
        for file in files:
            df=pd.read_csv("/data/watermarkingTraj/data/All_results/TrajGuard/our_data/data/"+file,header=None)
            df.columns=['traj_id','time','latitude','longitude']
            
            trip_id=re.split("\.",file)[0]
            pd.options.mode.chained_assignment = None
            df["time"]=df['time'].astype('datetime64[ns]') 
            
            df['time_diff'] = df["time"].diff().apply(lambda x: x / np.timedelta64(1, 's')).fillna(0).astype('int64')
            df['cum_sum'] = df['time_diff'].cumsum()

            append_watermarked  = addwatermark(df,d,watermark)
            append_watermarked['dist'] = append_watermarked.apply(lambda row: haversine_np(row['watermarked_long'],
                                                                                    row['watermarked_lat'],
                                                            row['longitude'], row['latitude']), axis=1)


            append_watermarked.to_csv('../'+config['global']['global_fol']+config['global']['technique']+'/our_data/watermarked_data/'+trip_id+'.csv',index=False)
           watermarkExtract(append_watermarked,d,trip_id,watermark)
    elapsed_time_fl = (time.time() - start)
    print('elapsed_time_fl=',elapsed_time_fl)


