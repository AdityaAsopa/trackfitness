# Description: This script calculates the change in HRV value during an orthostatic HR measurement test.
# Test: The test is performed by sitting comfortably on a chair for 2 minutes, and then standing up for 2 minutes without moving.
# Input: The input is a CSV file with two columns: timestamp and RR-interval in milliseconds. Use Polar Sensor to record the RR-interval.
# Output: The output is the baseline HRV and end HRV values.
# The baseline HRV is the average HRV value during the first 60 seconds of the test, and the end HRV is the average HRV value during the last 60 seconds of the test.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

def rmssd(rr_intervals):
    """
    Calculate the root mean square of successive differences of RR-intervals
    
    Parameters:
    rr_intervals (array-like): Array of RR-intervals
    
    Returns:
    float: Root mean square of successive differences of RR-intervals
    """
    return np.round( np.sqrt(np.mean(np.diff(rr_intervals)**2)), 2)


def save_record(record_datetime, mean_pre_HR, max_oHR, mean_post_HR, baseline_hrv, end_hrv, flag, record_filepath="orthostatic_hrv_record.txt" ):
    """
    Save the baseline HRV and end HRV to a text file,
    where each row contains, date, time, baseline HRV, and end HRV
    
    Parameters:
    record_filepath (str): Path to the CSV file
    baseline_hrv (float): Baseline HRV
    end_hrv (float): End HRV
    
    Returns:
    None
    """
    # folder should be the path of the script
    folder = Path(sys.argv[0]).parent
    record_filepath = folder / record_filepath
    # check if the file exists, if not create it and append the values to new row
    try:
        with open(record_filepath, 'a') as f:
            f.write(f"{pd.to_datetime('now')}, {mean_pre_HR}, {max_oHR}, {mean_post_HR}, {baseline_hrv}, {end_hrv}, {flag}\n")
    except FileNotFoundError:
        with open(record_filepath, 'w') as f:
            f.write("date, preHR, maxHR, postHR, baselineHRV, endHRV, flag\n")
            f.write(f"{record_datetime}, {mean_pre_HR}, {max_oHR}, {mean_post_HR}, {baseline_hrv}, {end_hrv}, {flag}\n")
    

def plothr(df):
    # make a new plot figure from subplot mosaic
    fig, ax = plt.subplot_mosaic(
                    [["A", "C"],
                    ["B", "C"]],
                    layout="constrained", figsize=(12, 6))

    # ax = [axs, axs.twinx()]
    window_size = 5  # Define the window size for the moving average

    # Calculate the moving average of HR
    window = np.ones(window_size) / window_size
    HR_ma = np.convolve(df['HR'], window, mode='same')

    ax["A"].plot(df['Timestamp'], df['HR'], color=[177/256,60/256,108/256], label='Heart Rate (Moving Average)')
    ax["A"].plot(df['Timestamp'], HR_ma, linewidth=4, color=[227/256,104/256,92/256], )
    ax["A"].set_title('Heart Rate')
    ax["A"].set_ylabel('BPM')
    # add a vertical line at 'Timestamp' = 60 and -60
    ax["A"].axvline(x=60, color='grey', linestyle='--',)
    ax["A"].axvline(x=end_time-60, color='grey', linestyle='--',)    
    ax["A"].set_xlim(0, df['Timestamp'].values[-1])
    ax["A"].set_ylim(0, 120)
    ax["A"].text(0.1, 0.9,'Pre', fontsize=12, color='black', transform=ax["A"].transAxes,)
    ax["A"].text(0.9, 0.9, 'Post', fontsize=12, color='black', transform=ax["A"].transAxes,)

    RR_ma = np.convolve(df['RR'], window, mode='same')
    ax["B"].plot(df['Timestamp'], df['RR'], color=[64/256,151/256,182/256], label='RR-interval')
    ax["B"].plot(df['Timestamp'], RR_ma, linewidth=4, color=[169/256, 220/256, 164/256],)
    ax["B"].set_title('RR-interval')
    ax["B"].set_xlabel('Time (s)')
    ax["B"].set_ylabel('ms')
    # add a vertical line at 'Timestamp' = 60 and -60
    ax["B"].axvline(x=60, color='grey', linestyle='--',)
    ax["B"].axvline(x=end_time-60, color='grey', linestyle='--',) 
    # add text 'pre' and 'post'
    ax["B"].set_xlim(0, df['Timestamp'].values[-1])
    ax["B"].set_ylim(0, 1500)

    # remove spines from top and right
    ax["A"].spines['top'].set_visible(False)
    ax["A"].spines['bottom'].set_visible(False)
    ax["A"].spines['right'].set_visible(False)
    ax["B"].spines['top'].set_visible(False)
    ax["B"].spines['right'].set_visible(False)

    # for plot 'C' get the max HR timepoint after the baseline period and plot 30 data points around it
    dfslice = df[ (df['epoch'] == 'orthostatic') ]

    maxOHR = dfslice['HR'].max()
    # find the index of the maxOHR
    maxOHR_index = df[df['HR'] == maxOHR].index[0]
    # find the time of the maxOHR
    maxOHR_time = df['Timestamp'].iloc[maxOHR_index]
    print(f'Max HR: {maxOHR} at {maxOHR_time} seconds')

    # plot 'HR' from maxOHR_index-60 to maxOHR_index+60
    ax["C"].plot(df['Timestamp'], df['HR'])
    ax["C"].set_ylim(0, 200)
    ax["C"].set_xlim(maxOHR_time-15, maxOHR_time+15)

    ax["C"].spines['top'].set_visible(False)
    ax["C"].spines['right'].set_visible(False)

    plt.show()
    # save plot
    folder = Path(sys.argv[0]).parent
    
    print(record_datetime)
    fig_filepath = folder / str(f'orthostatic_hrv_test_{record_datetime}.png')
    fig.savefig(fig_filepath)


def main(filepath, plot=True):
    """
    Main function to calculate baseline HRV and end HRV from a CSV file
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    None
    """
    # read the file
    df = pd.read_csv(filepath, sep=';', header=0)
    df.rename(columns={'Phone timestamp': 'Timestamp'}, inplace=True)
    df.rename(columns={'RR-interval [ms]': 'RR'}, inplace=True)
    
    # save record_datetime as a global variable
    global record_datetime
    record_datetime = pd.to_datetime(df.iloc[0]['Timestamp']).strftime('%Y-%m-%d-%H-%M-%S')

    # first column is timestamp, second column is RR-interval
    # subtract the first timestamp from all the timestamps to get the time in seconds
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    global end_time
    end_time = df['Timestamp'].iloc[-1]

    df['HR'] = 60000 / df['RR']
    # make a column called 'epoch' which has three values: 'pre', 'orthostatic', 'post'
    df['epoch'] = np.nan
    df.loc[df['Timestamp'] < baseline_period, 'epoch'] = 'pre'
    df.loc[(df['Timestamp'] > baseline_period) & (df['Timestamp'] < end_time-baseline_period), 'epoch'] = 'orthostatic'
    df.loc[df['Timestamp'] > end_time-baseline_period, 'epoch'] = 'post'

    RR = df['RR'].values
    HR = df['HR'].values
    baseline_hrv = rmssd(df[df['epoch']=='pre']['RR'])
    end_hrv =  rmssd(df[df['epoch']=='post']['RR'])

    max_HR = np.round(np.max(HR), 0)
    # pre period
    mean_pre_HR = np.round(np.mean(df[df['epoch']=='pre']['HR']), 0)
    max_pre_HR = np.round(np.max(df[df['epoch']=='pre']['HR']), 0)
    # orthostatic period
    max_oHR = np.round(np.max(df[df['epoch']=='orthostatic']['HR']), 0)
    # post period
    mean_post_HR = np.round(np.mean(df[df['epoch']=='post']['HR']), 0)
    max_post_HR = np.round(np.max(df[df['epoch']=='post']['HR']), 0)

    # print using ascii symbols as a timeline
    print('Baseline HRV:', baseline_hrv)
    print('End HRV:', end_hrv)
    print('Max HR:', max_HR)
    print('Max HR during pre:', max_pre_HR)
    print('Mean HR during pre:', mean_pre_HR)
    print('Max HR during orthostatic:', max_oHR)
    print('Max HR during post:', max_post_HR)
    print('Mean HR during post:', mean_post_HR)

    flag= ''
    # check if the max of HR overall is higher than pre_max and post_max
    if max_oHR < max_post_HR:
        print('HR increased after standing up but has not come down. Are you sure you did not move?')
        print('Flagging the record')
        flag = 'post High'
    elif max_oHR < max_pre_HR:
        print('Baseline HR higher than standing HR: Are you sure you stood up? Or were you moving during baseline period?')
        print('Flagging the record')
        flag = 'pre High'
        
    if plot:
        plothr(df)
    print('Saving Record...')
    save_record(record_datetime, mean_pre_HR, max_oHR, mean_post_HR, baseline_hrv, end_hrv, flag)


if __name__ == "__main__":
    print('Running Orthostatic HRV Test...')
    filepath = sys.argv[1]
    global baseline_period
    if int(sys.argv[2])<2:
        print('Baseline period should be in seconds and greater than 60')
        sys.exit()
    else:
        baseline_period = int(sys.argv[2])
    main(filepath, plot=True) 