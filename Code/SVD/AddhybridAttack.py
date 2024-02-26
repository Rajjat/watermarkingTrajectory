import numpy as np
import pandas as pd
import time
from numpy import diag
from numpy import zeros
import os
import logging
import argparse
import configparser
import numpy as np
import sys

from math import radians, cos, sin, asin, sqrt

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
def remove_non_significant_bits(trace, spatial_decimals=2, temporal_decimals=2):
    rounded_trace = np.zeros_like(trace)

    rounded_trace[:, :2] = np.around(trace[:, :2], decimals=spatial_decimals)
    rounded_trace[:, 2:] = np.around(trace[:, 2:], decimals=temporal_decimals)

    return rounded_trace


def _add_noise_with_signal_to_noise_ratio(signal, signal_to_noise_ratio, indices=None):
    # Calculate signal power and convert to dB
    signal_average = np.mean(signal)
    signal_average_db = 10 * np.log10(signal_average)
    # Calculate noise according to [2] then convert to watts
    noise_average_db = signal_average_db - signal_to_noise_ratio
    noise_average = 10 ** (noise_average_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    if indices is None:
        noise = np.random.normal(mean_noise, np.sqrt(noise_average), len(signal))
        return signal + noise
    else:
        noisy_signal = np.zeros_like(signal)
        noisy_signal += signal
        noisy_signal[indices] += np.random.normal(mean_noise, np.sqrt(noise_average), np.sum(indices))
        return noisy_signal


def add_signal_noise(trace, signal_to_noise_ratio=105):
    noisy_trace = np.zeros_like(trace)

    # Noise up the original signal
    noisy_trace[:, 0] = _add_noise_with_signal_to_noise_ratio(trace[:, 0], signal_to_noise_ratio)
    noisy_trace[:, 1] = _add_noise_with_signal_to_noise_ratio(trace[:, 1], signal_to_noise_ratio)
    #noisy_trace[:, 2] = _add_noise_with_signal_to_noise_ratio(trace[:, 2], signal_to_noise_ratio)
    noisy_trace[:, 2] = trace[:, 2]


    return noisy_trace


def add_white_noise(trace, spatial_variance=0.00002, temporal_variance=0):
    noisy_trace = np.zeros_like(trace)

    number_of_samples = len(trace[:, 0])
    noisy_trace[:, 0] = trace[:, 0] + np.random.normal(0, spatial_variance, number_of_samples)
    noisy_trace[:, 1] = trace[:, 1] + np.random.normal(0, spatial_variance, number_of_samples)
    #noisy_trace[:, 2] = trace[:, 2]  # + np.random.normal(0, temporal_variance, number_of_samples)
    noisy_trace[:, 2] = trace[:, 2]

    return noisy_trace


def add_outliers_with_signal_to_noise_ratio(trace, signal_to_noise_ratio=90, affected_percentage=.03):
    number_of_samples = len(trace[:, 0])
    affected_indices = np.random.choice([False, True],
                                        size=(number_of_samples,),
                                        p=[1 - affected_percentage, affected_percentage])

    noisy_trace = np.zeros_like(trace)

    # Noise up the original signal
    noisy_trace[:, 0] = _add_noise_with_signal_to_noise_ratio(trace[:, 0],
                                                              signal_to_noise_ratio,
                                                              indices=affected_indices)
    noisy_trace[:, 1] = _add_noise_with_signal_to_noise_ratio(trace[:, 1],
                                                              signal_to_noise_ratio,
                                                              indices=affected_indices)
    #noisy_trace[:, 2] = _add_noise_with_signal_to_noise_ratio(trace[:, 2],
                                                            #  signal_to_noise_ratio,
                                                             # indices=affected_indices)
    noisy_trace[:, 2]=trace[:, 2]
    return noisy_trace


def remove_random_points(trace, removal_percentage=.005):
    number_of_samples = len(trace[:, 0])
    affected_indices = np.random.choice([True, False],
                                        size=(number_of_samples,),
                                        p=[removal_percentage, 1 - removal_percentage])
    noisy_trace = trace.copy()
    for i, replace in enumerate(affected_indices):
        if replace:
            if i == 0:
                noisy_trace[0] = noisy_trace[1]
            else:
                noisy_trace[i] = noisy_trace[i - 1]
    return noisy_trace


def remove_random_points_with_path(trace, removal_percentage=.01): 
    #print(trace)#1
    trace=trace[:,0:2]
    #print(trace)
    number_of_samples = len(trace[:, 0])
    affected_indices = np.random.choice([True, False],
                                        size=(number_of_samples,),
                                        p=[removal_percentage, 1 - removal_percentage])
    affected_indices[0] = False
    affected_indices[-1] = False

    noisy_trace = trace.copy()
    first_index_replace = None
    last_skeleton_point = trace[0]
    for i, replace in enumerate(affected_indices):
        if replace and first_index_replace is None:
            first_index_replace = i
        elif not replace:
            if first_index_replace is not None:
                for replace_index in range(first_index_replace, i):
                    noisy_trace[replace_index] = (replace_index - first_index_replace + 1) \
                                                 / (i - first_index_replace + 1) * (
                                                         trace[i] - last_skeleton_point) + last_skeleton_point

            last_skeleton_point = trace[i]
            first_index_replace = None

    return noisy_trace


def replace_non_skeleton_points_with_start_point(trace, epsilon=.00001):
    import rdp
    trace=trace[:,0:2]
    #noisy_trace = trace.copy()
    noisy_trace = trace.copy()

    mask = rdp.rdp(trace, epsilon, return_mask=True)

    for i, keep in enumerate(mask):
        if not keep:
            noisy_trace[i] = noisy_trace[i - 1]
    return noisy_trace


def replace_non_skeleton_points_with_path(trace, epsilon=.00005):
    import rdp
    trace=trace[:,0:2]
    noisy_trace = trace.copy()

    mask = rdp.rdp(trace, epsilon, return_mask=True)

    last_skeleton_point = trace[0]
    first_index_replace = None
    for i, keep in enumerate(mask):
        if not keep and first_index_replace is None:
            first_index_replace = i
        elif keep:
            if first_index_replace is not None:
                for replace_index in range(first_index_replace, i):
                    noisy_trace[replace_index] = (replace_index - first_index_replace + 1) \
                                                 / (i - first_index_replace + 1) * (
                                                         trace[i] - last_skeleton_point) + last_skeleton_point

            last_skeleton_point = trace[i]
            first_index_replace = None

    return noisy_trace


# helper function for resampling
# based on https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy/54628145
def moving_average(a, n=3):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / na


def resample_trace_along_path(trace, avergaging_window=3):
    noisy_trace = trace.copy()

    averaged_trace = moving_average(trace, avergaging_window)
    noisy_trace[:-len(averaged_trace)] = averaged_trace

    return noisy_trace
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
    i=0
    start = time.time()
    noise='hybrid'
    directory = os.path.join("/data/watermarkingTraj/data/All_results/SVD/our_data/noise_traj/cropfromlast/")
    for root,dirs,files in os.walk(directory):
        for file in files:
            df=pd.read_csv("/data/watermarkingTraj/data/All_results/SVD/our_data/noise_traj/cropfromlast/"+file,header=0)
             data_three = df[['watermarked_lat', 'watermarked_long','cum_sum','latitude','longitude']]
            trip_id=df.traj_id.values[0]
            noisy_trace = add_white_noise(data_three.values)
            noisy_trace1=remove_random_points(noisy_trace)

            df_noisy = pd.DataFrame(noisy_trace1)
            df_noisy=df_noisy.iloc[:,0:2]
            df_noisy.columns=['c_w', 'c_w_long']

            traj_folder = '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'
            traj_folder = traj_folder + "/"
            final_folder=traj_folder
            final_folder = final_folder + "/"
            path1=final_folder + noise
            if not os.path.isdir(path1):
                os.mkdir(path1)

            compare_dist=df[['watermarked_lat','watermarked_long','time']]
            frames=[df_noisy,compare_dist]
            final_df=pd.concat(frames, axis=1)
            final_df['dist']=final_df.apply(lambda row: haversine(row['watermarked_long'],row['watermarked_lat'],
                                                                  row['c_w_long'],row['c_w']), axis=1)
            final_df['trip_id']=trip_id
            final_df.to_csv( '../'+config['global']['global_fol']+config['global']['technique']+'/our_data/noise_traj/'+noise+'/'+str(trip_id)+'.csv',index=False,header=True)
elapsed_time_fl = (time.time() - start)
print('elapsed_time_fl=',elapsed_time_fl)



