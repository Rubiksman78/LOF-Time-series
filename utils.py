import time
import numpy as np
import datetime

def turn_df_to_df_with_value(df):
    df_with_value = df.copy()
    df_with_value['value'] = np.sqrt(np.square(df_with_value['valence']) \
        + np.square(df_with_value['arousal']))
    df_with_value['time_stamp'] = df['sent_at'].apply(
        lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())) 
    df_with_value.time_stamp -= df_with_value.time_stamp.min()
    return df_with_value

def min_max_scaling(X:np.ndarray, a:float, b:float) -> np.ndarray:
    X_min = np.min(X)
    X_max = np.max(X)
    scaled_array = a + (X-X_min)*(b-a)/(X_max-X_min)
    return scaled_array

def plot_outliers(points,outliers,lof_scores,time):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(time,points[:,0],'b.')
    ax1.set_xlabel('time stamp')
    ax1.set_ylabel('value')
    ax1.plot(time[outliers],points[outliers,0],'ro')
    ax2.plot(time,lof_scores,'b.')
    ax2.plot(time[outliers],lof_scores[outliers],'ro')
    ax2.set_xlabel('time stamp')
    ax2.set_ylabel('LOF score')
    plt.show()