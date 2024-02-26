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
    c = []
    lp=int(config['global']['slice']) 
    for i in range(lp):
        c.append(complex(lat_array[i][0], lat_array[i][1]))

    x1 = fft(c)
    return x1


def watermarkExtract(watermark,x1,data_wc):
    lp=int(config['global']['slice']) 
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

    lp=int(config['global']['slice']) 
    df1 = pd.read_csv(
        '/data/watermarkingTraj/data/All_results/'+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/watermark_corrWithDistance.csv',header=None)    
    print(df1)
    df1.columns=['trip_id','mean_dist','min_dist','max_dist','watermark_corr']
    trip_idSeries = df1['trip_id'].values
    noises=['remove_random_points_with_path','add_outliers_with_signal_to_noise_ratio','add_signal_noise',
       'add_white_noise','replace_non_skeleton_points_with_path',
       'remove_random_points','hybrid','double_emded','cropfromlast']
    df_concat1=pd.DataFrame(columns=['traj_id','Noise','corr_value','avg','min','max'])
    for trip_id in trip_idSeries:
        for noise in noises:
            print("*********"+noise+"***********")
            df_noise = pd.DataFrame(columns=['traj_id','Noise','corr_value','avg','min','max'])
            watermark_256=[]
            extracted_watermark_256=[]
            corr_watermark_arr=[]
            df_2 = pd.read_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/noise_added/'+noise+'.csv')

            for i in range(16): 
                watermark=np.load("/data/watermarkingTraj/data/All_results/FFT_complexNum/256_len/"+config['global']['slicenumber']+"/payload_strength/different_watermarks/payload_5/watermark_5.npy",allow_pickle=True) 
                watermark=watermark[0:16]
                watermark_256.append(watermark)
                df_3=df_2[lp*i:lp*i+lp]
                df_3=df_3.reset_index(drop=True)
                x1  = addwatermark(df_3)


                extract_wc2=watermarkExtract(watermark,x1,df_3)
                extracted_watermark_256.append(extract_wc2)

                corr_watermark = ncc(np.array(extract_wc2).flatten(), np.array(watermark).flatten())
                corr_watermark_arr.append(corr_watermark)

            corr_watermark2=statistics.mean(corr_watermark_arr)
            df_2['dist'] = df_2.apply(lambda row: haversine(row['c_w_long'], row['c_w'],
                                                                    row['watermarked_long'], row['watermarked_lat']), axis=1)

            avg = df_2["dist"].mean()
            max1 = df_2["dist"].max()
            min1 = df_2["dist"].min()                                                                               
            df_noise = df_noise.append({'traj_id': trip_id,'Noise':noise,'corr_value':"{0:0.4f}".format(corr_watermark2),'avg':"{0:0.4f}".format(avg),'max':"{0:0.4f}".format(max1),'min':"{0:0.4f}".format(min1)}, ignore_index=True)
            df_noise.to_csv('../'+config['global']['global_fol']+config['global']['technique']+'/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/'+trip_id+'/watermarking/watermark_cor_withNoises_'+noise+'.csv',index=False)

            df_concat1 = pd.concat([df_concat1, df_noise])
    df_concat1.to_csv('../'+config['global']['global_fol'] + config['global']['technique'] + '/256_len/'+config['global']['slicenumber']+'/payload_strength/different_watermarks/payload_5/watermarked_data/watermark_cor_withNoises_all.csv',
                index=False)

