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
# import kruskal and KS test from scipy
from scipy.stats import kruskal, ks_2samp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import fitparse
# import seaborn as sns

def fit_to_df(filepath):
    fitfile = fitparse.FitFile(filepath)
    data = []
    for record in fitfile.get_messages('record'):
        r = {}
        for datax in record:
            r[datax.name] = datax.value
        data.append(r)
    return pd.DataFrame(data)

def tcx_to_df(filepath):
    ''' Read TCX file and return a dataframe '''
    '''
    <?xml version="1.0"?>
    <TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:ns3="http://www.garmin.com/xmlschemas/ActivityExtension/v2" xsi:schemaLocation="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2 http://www.garmin.com/xmlschemas/TrainingCenterDatabasev2.xsd">
    '''
    from xml.etree import ElementTree
    tree = ElementTree.parse(filepath)
    root = tree.getroot()
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    activities = root.findall('tcx:Activities/tcx:Activity', ns)
    data = []
    for activity in activities:
        lap = activity.find('tcx:Lap', ns)
        track = lap.find('tcx:Track', ns)
        for trackpoint in track.findall('tcx:Trackpoint', ns):
            r = {}
            for child in trackpoint:
                if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time':
                    r['Timestamp'] = child.text
                if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}HeartRateBpm':
                    r['HR'] = child.find('tcx:Value', ns).text
                
            data.append(r)
    return pd.DataFrame(data)

def rr_to_rmssd(rr_intervals_ms):
    """
    Calculate the root mean square of successive differences of RR-intervals
    
    Parameters:
    rr_intervals (array-like): Array of RR-intervals (in milliseconds)
    
    Returns:
    float: Root mean square of successive differences of RR-intervals
    """
    return np.round( np.sqrt(np.median(np.diff(rr_intervals_ms)**2)), 2)

def hr_to_rmssd(hr_bpm):
    """
    Calculate the root mean square of successive differences of HR
    
    Parameters:
    hr (array-like): Array of HR
    
    Returns:
    float: Root mean square of successive differences of HR
    """
    return rr_to_rmssd(60000/hr_bpm)

def rr_to_sdnn(rr_intervals_ms):
    """
    Calculate the standard deviation of RR-intervals
    
    Parameters:
    rr_intervals (array-like): Array of RR-intervals (in milliseconds)
    
    Returns:
    float: Standard deviation of RR-intervals
    """
    return np.round(np.std(rr_intervals_ms), 2)

def hr_to_sdnn(hr_bpm):
    """
    Calculate the standard deviation of HR
    
    Parameters:
    hr (array-like): Array of HR
    
    Returns:
    float: Standard deviation of HR
    """
    return rr_to_sdnn(60000/hr_bpm)

def HRrecovery(df, fit_window=60, xmin=0, plot=True, fig='', axx=''):
    # get the time it take to reach the maximum HR: 10-90% of the maximum HR
    # first get the baseline corrected HR
    hrbaseline = df[df['epoch']=='pre']['HR'].median()
    df['HR_baseline_corrected'] = df['HR'] - hrbaseline

    # what is the last 'Timestamp' value when HR was 10% of maxHR before the maximum HR
    maxHR = df[ (df['epoch']=='orthostatic') ][ 'HR_baseline_corrected' ].max()
    ssHR = df[ (df['epoch']=='post') ][ 'HR_baseline_corrected' ].median()
    endOrthoHR = df[ (df['epoch']=='orthostatic') ][ 'HR_baseline_corrected' ].values[-1]
    time_max = df[(df['HR_baseline_corrected']==maxHR)]['Timestamp'].values[0]

    # get 'Timestamp' value of all those rows where HR is 10% of maxHR
    time_10_pre = df[(df['HR_baseline_corrected']<maxHR*0.1) & (df['Timestamp']<time_max)]['Timestamp'].values[-1]
    time_90_pre = df[(df['HR_baseline_corrected']<maxHR*0.9) & (df['Timestamp']<time_max)]['Timestamp'].values[-1]
    rise_slope = (maxHR*0.9 - maxHR*0.1) / (time_90_pre - time_10_pre)

    dfslice = df[ df['Timestamp'] > time_max ]
    T = dfslice['Timestamp']-time_max

    # # for the fall phase we will fit an exponential to the falling HR
    # # first we will get the HR values for the fall phase
    dfslice2 = dfslice[ (dfslice['Timestamp'] < time_max+fit_window) ]
    time = dfslice2['Timestamp'].values - time_max
    hr = dfslice2['HR_baseline_corrected'].values

    def fixed_exponential(t, a, tau, ss):
        return a * np.exp(-1 * t/tau) + ss

    print('maxHR:', maxHR, 'endOrthoHR:', endOrthoHR, 'ssHR:', ssHR)
    popt, _ = curve_fit(fixed_exponential, time, hr, p0=(maxHR, 15, endOrthoHR), bounds=([0.99*(maxHR-endOrthoHR),5,0], [1.2*maxHR, 30, endOrthoHR]))
    # popt, _ = curve_fit(fixed_exponential, time, hr, p0=(maxHR, 15, endOrthoHR), bounds=([0.9*(maxHR-endOrthoHR),5,0], [1.1*(maxHR+endOrthoHR), 30, endOrthoHR]))
    a, tau, ss = popt
    # # plot the fit
    HRR60 = np.round(maxHR - fixed_exponential(60, *popt))
    print(f'delta HR by the end of 60s: {HRR60}')

    if plot==False:
        return rise_slope, tau, ss, maxHR, HRR60, None, None

    if not axx:
        fig, axx = plt.subplots(1, 1, figsize=(12, 6))
    
    # ax.plot(dfslice['Timestamp'], dfslice['HR_baseline_corrected']) #--time_max+xmin
    axx.plot([time_10_pre, time_90_pre], hrbaseline+[maxHR*0.1, maxHR*0.9], '-o', color='orange',  label=f'HR Rise: {rise_slope:.1f} bpm/s')
    axx.plot(T+xmin, hrbaseline+fixed_exponential(T, a, tau, ss), '-g', label=f'60s HR Recovery: {HRR60} bpm')
    
    return rise_slope, tau, ss, a, HRR60, fig, axx


def save_record(record_datetime, params, record_filepath="orthostatic_hrv_record.txt" ):
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
    record_filepath = folder / record_filepath

    dfparams = pd.DataFrame(params, index=[0])

    dfsave = pd.read_csv(record_filepath) if Path(record_filepath).exists() else pd.DataFrame()
    dfsave = pd.concat([dfsave, dfparams], ignore_index=True)

    dfsave.to_csv(record_filepath, index=False)
    

    print(f'Record saved to {record_filepath}')
    

def plothr(df, pre_baseline_period=60, post_baseline_period=60, plot_window_size=60, end_time=''):
    # make a new plot figure from subplot mosaic
    fig, ax = plt.subplot_mosaic(
                    [["A", "C"],
                    ["B", "C"]],
                    layout="constrained", figsize=(12, 6))

    # ax = [axs, axs.twinx()]
    window_size = 5  # Define the window size for the moving average
    if not end_time:
        end_time = df['Timestamp'].values[-1]
    # Calculate the moving average of HR
    window = np.ones(window_size) / window_size
    HR_ma = np.convolve(df['HR'], window, mode='same')

    ax["A"].plot(df['Timestamp'], df['HR'], color=[177/256,60/256,108/256], label='Heart Rate (Moving Average)')
    ax["A"].plot(df['Timestamp'], HR_ma, linewidth=4, color=[227/256,104/256,92/256], )
    ax["A"].set_title('Heart Rate')
    ax["A"].set_ylabel('BPM')
    # add a vertical line at 'Timestamp' = pre_baseline_period and -post_baseline_period
    ax["A"].axvline(x=pre_baseline_period, color='grey', linestyle='--',)
    ax["A"].axvline(x=end_time-post_baseline_period, color='grey', linestyle='--',)    
    ax["A"].set_xlim(0, df['Timestamp'].values[-1])
    ax["A"].set_ylim(40, 120)
    ax["A"].text(0.1, 0.9,'Pre', fontsize=12, color='black', transform=ax["A"].transAxes,)
    ax["A"].text(0.9, 0.9, 'Post', fontsize=12, color='black', transform=ax["A"].transAxes,)

    RR_ma = np.convolve(df['RR'], window, mode='same')
    ax["B"].plot(df['Timestamp'], df['RR'], color=[64/256,151/256,182/256], label='RR-interval')
    ax["B"].plot(df['Timestamp'], RR_ma, linewidth=4, color=[169/256, 220/256, 164/256],)
    ax["B"].set_title('RR-interval')
    ax["B"].set_xlabel('Time (s)')
    ax["B"].set_ylabel('ms')
    # add a vertical line at 'Timestamp' = pre_baseline_period and -post_baseline_period
    ax["B"].axvline(x=pre_baseline_period, color='grey', linestyle='--',)
    ax["B"].axvline(x=end_time-post_baseline_period, color='grey', linestyle='--',) 
    # add text 'pre' and 'post'
    ax["B"].set_xlim(0, df['Timestamp'].values[-1])
    ax["B"].set_ylim(500, 1500)

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
    ax["C"].plot(df['Timestamp'], df['HR'], linewidth=2, color=[177/256,60/256,108/256], label='HR')    
    ax["C"].set_ylim(40, 120)
    ax["C"].set_xlim(maxOHR_time-plot_window_size, maxOHR_time+plot_window_size)
    # add horizontal line at median of preHR and postHR
    ax["C"].axhline(y=df[df['epoch']=='pre']['HR'].median(), color='grey', linestyle='--',)
    ax["C"].axhline(y=df[df['epoch']=='post']['HR'].median(), color='grey', linestyle='--',)
    ax["C"].set_title('Orthostatic Dynamics of Heart Rate')
    ax["C"].set_xlabel('Time (s)')
    ax["C"].set_ylabel('BPM')
    ax["C"].spines['top'].set_visible(False)
    ax["C"].spines['right'].set_visible(False)


    rise_slope, tau, ss, maxHR, HRR60, fig, ax["C"] = HRrecovery(df, xmin=maxOHR_time, fig=fig, axx=ax["C"])
    ax["C"].legend()

    plt.show()
    
    print(record_datetime)
    fig_filepath = folder / str(f'orthostatic_hrv_test_{record_datetime}.png')
    print(fig_filepath)
    fig.savefig(fig_filepath)

    return rise_slope, tau, ss, maxHR, HRR60, fig, ax


def parse_workout_file(filepath, origin='Polar Sensor Logger Export'):
    ''' 
    origins = ['Polar Sensor Logger Export', 'Runalyze Fit Export' , 'Runalyze TCX Export']
    origin = origins[1]
    '''
    if origin == 'Polar Sensor Logger Export':
        df = pd.read_csv(filepath, sep=';', )
        df.rename(columns={'Phone timestamp': 'Timestamp'}, inplace=True)
        df.rename(columns={'RR-interval [ms]': 'RR'}, inplace=True)
        df['HR'] = 60000 / df['RR']

    elif origin == 'Runalyze Fit Export':
        df = fit_to_df(str(filepath))
        df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
        df.rename(columns={'heart_rate': 'HR'}, inplace=True)
        df['RR'] = 60000 / df['HR']    

    elif origin == 'Runalyze TCX Export':
        df = tcx_to_df(filepath)
        df['RR'] = 60000 / df['HR'] 

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.head(10)

    return df


def interpolate_cycle(cycle):
    # Create a function that interpolates the cycle
    f = interp1d(np.linspace(0, 1, len(cycle)), cycle, kind='linear')

    # Use the function to interpolate the cycle to 15 data points
    return f(np.linspace(0, 1, 15))


def breathanlyse(df):
    numsamples = df.shape[0]
    Fs = numsamples / df['Timestamp'].values[-1]

    # spectral analysis of 'pre' period 'HR'
    # find outa what frequencies of HR are seen in the data by doing a fourier or spectral anaylsis
    hr = df[df['epoch']=='pre']['HR'].values
    t = df[df['epoch']=='pre']['Timestamp'].values

    # one figure, with 2 subplots one cartesian and one polar
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(121, )
    # make the ax2 polar projection
    ax2 = plt.subplot(122, polar=True)

    locs, _ = find_peaks(hr, prominence=3)
    resp = 60 / np.mean(np.diff(t[locs])) # per min
    cycle_time = np.mean(np.diff(t[locs])) # in seconds
    cycle_time_std = np.std(np.diff(t[locs])) # in seconds


    resp_cycles = []
    for i, tt in enumerate(locs[:-1]):
        hrslice = hr[tt: locs[i+1]]
        resp_cycles.append(hrslice)
        
    for i, cycle in enumerate(resp_cycles):
        # Create a list of angles based on the number of measurements in the cycle
        angles = np.linspace(0, 2 * np.pi, len(cycle)) 
        ax1.plot(angles, cycle, color=np.array([72,34,115])/255, linewidth=1, alpha=0.5)

    # also plot an average respiratory cycle - HR response
    # Apply the function to each cycle in your data
    interpolated_data = np.array([interpolate_cycle(cycle) for cycle in resp_cycles])
    # Calculate the average heart rate for each interpolated angle
    avg_hr = np.mean(interpolated_data, axis=0)
    angles = np.linspace(0, 2 * np.pi, 15) 
    ax1.plot(angles, avg_hr, color='k', linewidth=3)
    # Show the plots
    lower_limit, upper_limit = 40, 80
    ax1.set_ylim([lower_limit, upper_limit])
    # Shade the inhalation phase
    [start_inhalation, end_inhalation] = 0, np.pi
    [start_exhalation, end_exhalation] = np.pi, 2*np.pi

    # # Replace start_inhalation and end_inhalation with the start and end angles of the inhalation phase
    ax1.fill_between(np.linspace(0,np.pi,180), np.linspace(40,40,180), np.linspace(80,80,180), color=np.array([37,134,142])/255, alpha=0.5)
    # # Shade the exhalation phase
    ax1.fill_between(np.linspace(np.pi,2*np.pi,180), np.linspace(40,40,180), np.linspace(80,80,180), color=np.array([164,49,127])/255, alpha=0.5)


    # make grid white
    ax1.yaxis.grid(color='white')
    ax1.xaxis.grid(color='white')
    # change the tick labels
    ax1.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False), labels=['Start Inhale', '','Start Exhale',''])
    ax1.set_yticks(np.linspace(40, 80, 5), labels=['40', '50', '60', '70', '80'])
    ax1.set_xlim([0, 2*np.pi])
    # remove spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)



    # make ax2 polar
    # ax2.set_projection('polar')
    ax2.set_theta_zero_location("N")
    # Assuming `data` is your list of vectors, where each vector is heart rate during one breathing cycle

    for i, cycle in enumerate(resp_cycles):
        # Create a list of angles based on the number of measurements in the cycle
        angles = np.linspace(0, 2 * np.pi, len(cycle)) 
        ax2.plot(angles, cycle, color=np.array([72,34,115])/255, linewidth=1, alpha=0.5)

    # also plot an average respiratory cycle - HR response
    # Apply the function to each cycle in your data
    interpolated_data = np.array([interpolate_cycle(cycle) for cycle in resp_cycles])
    # Calculate the average heart rate for each interpolated angle
    avg_hr = np.mean(interpolated_data, axis=0)
    angles = np.linspace(0, 2 * np.pi, 15) 
    ax2.plot(angles, avg_hr, color='k', linewidth=3)
    # Show the plots
    lower_limit, upper_limit = 40, 80
    ax2.set_ylim([lower_limit, upper_limit])
    # Shade the inhalation phase
    [start_inhalation, end_inhalation] = 0, np.pi
    [start_exhalation, end_exhalation] = np.pi, 2*np.pi

    # # Replace start_inhalation and end_inhalation with the start and end angles of the inhalation phase
    ax2.fill_between(np.linspace(0,np.pi,180), np.linspace(40,40,180), np.linspace(80,80,180), color=np.array([37,134,142])/255, alpha=0.5)
    # # Shade the exhalation phase
    ax2.fill_between(np.linspace(np.pi,2*np.pi,180), np.linspace(40,40,180), np.linspace(80,80,180), color=np.array([164,49,127])/255, alpha=0.5)
    # remove outer circle
    ax2.spines['polar'].set_visible(False)
    # make grid white
    ax2.yaxis.grid(color='white')
    ax2.xaxis.grid(color='white')
    # change the tick labels
    ax2.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False), labels=['Start Inhale', '','Start Exhale',''])
    ax2.set_yticks(np.linspace(40, 80, 5), labels=['40', '50', '60', '70', '80'])
    # change the location of ytick labels

    # save figure
    fig_filepath = folder / str(f'HR_response_to_breathing_{record_datetime}.png')
    print(fig_filepath)
    fig.savefig(fig_filepath)

    return resp, cycle_time, cycle_time_std


def main(filepath, origin, pre_baseline_period=60, post_baseline_period=60, plot=True):
    """
    Main function to calculate baseline HRV and end HRV from a CSV file
    
    Parameters:
    df (str): Dataframe with two columns: timestamp and RR-interval in milliseconds
    
    Returns:
    None
    """
    df = parse_workout_file(filepath, origin=origin)
    global folder
    folder = Path(filepath).parent

    # assert that the dataframe has correct columns: 'Timestamp', 'RR', 'HR'
    assert 'Timestamp' in df.columns, "Timestamp column not found"
    assert 'RR' in df.columns, "RR column not found"
    assert 'HR' in df.columns, "HR column not found"

    # save record_datetime as a global variable
    global record_datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    record_datetime = pd.to_datetime(df['Timestamp'].iloc[0]).strftime('%Y-%m-%d-%H-%M-%S')
    df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()


    # first column is timestamp, second column is RR-interval
    # subtract the first timestamp from all the timestamps to get the time in seconds
    
    global end_time
    end_time = df['Timestamp'].iloc[-1]   
    
    # make a column called 'epoch' which has three values: 'pre', 'orthostatic', 'post'
    df['epoch'] = ''
    df.loc[df['Timestamp'] <= pre_baseline_period, 'epoch'] = 'pre'
    df.loc[(df['Timestamp'] > pre_baseline_period) & (df['Timestamp'] <= end_time-post_baseline_period), 'epoch'] = 'orthostatic'
    df.loc[df['Timestamp'] > end_time-post_baseline_period, 'epoch'] = 'post'

    RR = df['RR'].values
    HR = df['HR'].values
    baseline_hrv = rr_to_rmssd(df[df['epoch']=='pre']['RR'])
    end_hrv =  rr_to_rmssd(df[df['epoch']=='post']['RR'])

    baseline_sdnn = rr_to_sdnn(df[df['epoch']=='pre']['RR'])
    end_sdnn = rr_to_sdnn(df[df['epoch']=='post']['RR'])

    max_HR = np.round(np.max(HR), 0)
    # pre period
    median_pre_HR = np.round(np.median(df[df['epoch']=='pre']['HR']), 0)
    max_pre_HR = np.round(np.max(df[df['epoch']=='pre']['HR']), 0)
    # orthostatic period
    max_oHR = np.round(np.max(df[df['epoch']=='orthostatic']['HR']), 0)
    # post period
    median_post_HR = np.round(np.median(df[df['epoch']=='post']['HR']), 0)
    max_post_HR = np.round(np.max(df[df['epoch']=='post']['HR']), 0)


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
        
    # check if pre_HR and post_HR are statistically different by running a kolmogorov smirnoff test
    preHR = df[df['epoch']=='pre']['HR']
    postHR = df[df['epoch']=='post']['HR']
    
    # run the KS test
    ks_stat, ks_pval = ks_2samp(preHR, postHR)
    # check if the p-value is less than 0.05
    if ks_pval < 0.05:
        print('HR during pre and post periods are statistically different. If PostHR is higher, You are cleared for exercise today.')
    else:
        print('HR during pre and post periods are not statistically different. This means that you are not cleared for exercise today. You need to recover.')

    # save all the params into a dict
    params = {
        'Timestamp': record_datetime,
        'Pre HRV (RMSSD)': np.round(baseline_hrv,1),
        'Post HRV (RMSSD)': np.round(end_hrv,1),
        'Pre HRV (SDNN)': np.round(baseline_sdnn,1),
        'Post HRV (SDNN)': np.round(end_sdnn,1),
        'max HR': np.round(max_HR,1),
        'Pre max HR': np.round(max_pre_HR,1),
        'Pre median HR': np.round(median_pre_HR,1),
        'Ortho max HR': np.round(max_oHR,1),
        'Post max HR': np.round(max_post_HR,1),
        'Post median HR': np.round(median_post_HR,1),
        'Flag': flag,
        'HRV Pre-Post KS-Stat': np.round(ks_stat,1),
        'HRV Pre-Post KS-pval': np.round(ks_pval,2)
    }

    if plot:
        rise_slope, tau, ss, maxHR, HRR60, fig, ax = plothr(df, pre_baseline_period=pre_baseline_period, post_baseline_period=post_baseline_period, plot_window_size=60,)
    else:
        print('No plot requested')
        rise_slope, tau, ss, maxHR, HRR60, fig, ax = HRrecovery(df, plot=False)

    # check breathing rate during pre period by doing a freq analysis
    resp, cycle_time, cycle_time_std = breathanlyse(df)


    # add the HR recovery parameters to the params dict
    params['Ortho HR Rise Slope'] = np.round(rise_slope,2)
    params['Ortho HR Fall Tau'] = np.round(tau,2)
    params['Ortho HR Fall HRinf'] = np.round(ss,2)
    params['maxHR'] = np.round(maxHR,2)
    params['HR Recovery 60s'] = np.round(HRR60,2)
    params['Breathing Rate'] = np.round(resp,2)
    params['Breathing Cycle Time'] = np.round(cycle_time,2)
    params['Breathing Cycle Fluctuation'] = np.round(cycle_time_std,2)


    print('Saving Record...')
    save_record(record_datetime, params)

    # Print all the params
    for key, value in params.items():
        print(f'{key}: {value}')


    return df, params, [fig, ax]


if __name__ == "__main__":
    print('Running Orthostatic HRV Test...')
    filepath = sys.argv[1]
    pre_baseline_period = int(sys.argv[2])
    post_baseline_period = int(sys.argv[3])
    
    main(filepath, origin='Polar Sensor Logger Export', pre_baseline_period=pre_baseline_period, post_baseline_period=post_baseline_period, plot=True) 