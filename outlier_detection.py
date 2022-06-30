import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import time
from config import DEFAULT_CONFIG
from utils import * 

#This function takes a dataframe and groups it by mean value every 3 minues
def mean_every_3_minutes(df):
    df_mean = df.groupby(df.time_stamp//time_step).mean()
    df_std = df.groupby(df.time_stamp//time_step).agg(
                      {'value':'std','time_stamp':'first'})
    return df_mean,df_std
   
def LOF(points:np.ndarray, k:int) -> np.ndarray:
    n = len(points)
    distMat = np.linalg.norm(points - points[:,None], axis = -1)
    kFirstNeib = np.argsort(distMat,axis=-1)[:,:k]
    kFirstDist = np.take_along_axis(distMat, kFirstNeib, axis=-1)                                              
    kD = kFirstDist[:,-1]
    RD = np.where(distMat>kD,distMat,kD)
    RDmean = np.mean(RD, 1)
    LRD = 1/RDmean
    LRDkFirst = np.array([np.mean(np.take_along_axis(LRD, kFirstNeib[i,:], axis=-1),-1) for i in range(n)]) 
    lof = LRDkFirst/LRD
    return lof

def detect_outliers(df,threshold,use_mean=True):
    df_with_value = turn_df_to_df_with_value(df)
    df_mean,df_std = mean_every_3_minutes(df_with_value)
    time = df_mean[["time_stamp"]]
    df_mean = df_mean[["value"]]
    df_std = df_std[["value"]]
    if use_mean:
        points = df_mean.values
    else: 
        points = df_std.values
    points[:,0] = min_max_scaling(points[:,0],0,1)
    lof_scores = LOF(points, k=k)
    outliers = np.where(lof_scores > threshold)[0]
    return points,outliers,lof_scores,time.values

if __name__ == "__main__":
    k = DEFAULT_CONFIG['k']
    threshold = DEFAULT_CONFIG['threshold']
    time_step = DEFAULT_CONFIG['time_step']
    use_mean = DEFAULT_CONFIG['use_mean']
    data_path = DEFAULT_CONFIG['data_path']
    df = pd.read_csv(data_path)
    points,outliers,lof_scores,time = detect_outliers(df,threshold,use_mean)
    outlier_times = np.concatenate(time[outliers])
    print(outlier_times)
    plot_outliers(points,outliers,lof_scores,time)