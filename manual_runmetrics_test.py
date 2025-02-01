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

def extract_trackpoint_params(trackpoint):
    # from xml.etree import ElementTree
    # tree = ElementTree.parse(filepath)
    # root = tree.getroot()
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    params = {}
    for child in trackpoint:
        '''all the childrend in tcx files:
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time 2024-05-27T18:16:24+05:30
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position 
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}AltitudeMeters 936
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Cadence 71
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters 0
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}HeartRateBpm 
            {http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Extensions 
        '''
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Time':
            params['Timestamp'] = child.text
            # convert 'Timestamp' to datetime object
            params['Timestamp'] = pd.to_datetime(params['Timestamp'])
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}HeartRateBpm':
            params['HR'] = child.find('tcx:Value', ns).text
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}DistanceMeters':
            params['Distance'] = child.text
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}AltitudeMeters':
            params['Altitude'] = child.text
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Cadence':
            params['Cadence'] = 2*int(child.text)
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Position':
            # position has two children: LatitudeDegrees and LongitudeDegrees
            params['Latitude'] = child.find('tcx:LatitudeDegrees', ns).text
            params['Longitude'] = child.find('tcx:LongitudeDegrees', ns).text
        # if any more info is there in extension tag
        if child.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Extensions':
            # extension tags can be: 'Speed', 'Watts', 'RunCadence', 'Temperature'
            for ext in child:
                if ext.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Speed':
                    params['Speed'] = ext.text
                if ext.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Watts':
                    params['Watts'] = ext.text
                if ext.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}RunCadence':
                    params['RunCadence'] = ext.text
                if ext.tag == '{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Temperature':
                    params['Temperature'] = ext.text

    return params


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
        # find all laps
        laps = activity.findall('tcx:Lap', ns)
        for lap_num, lap in enumerate(laps):
            lap_start_time = pd.to_datetime(lap.get('StartTime'))
            # find all tracks
            tracks = lap.findall('tcx:Track', ns)
            lap_intensity = lap.find('tcx:Intensity', ns).text
            for track_num, track in enumerate(tracks):
                # find all trackpoints
                trackpoints = track.findall('tcx:Trackpoint', ns)
                for trackpoint_num, trackpoint in enumerate(trackpoints):
                    r = extract_trackpoint_params(trackpoint)
                    r['Lap'] = lap_num
                    r['Track'] = track_num
                    r['Trackpoint'] = trackpoint_num
                    r['LapStartTime'] = lap_start_time
                    LapTime = r['Timestamp'] - lap_start_time
                    # convert LapTime to min:seconds
                    r['LapTime'] = LapTime.seconds/60
                    r['Epoch'] = lap_intensity                
                    
                    data.append(r)
    return pd.DataFrame(data) 


filepath = Path(sys.argv[1])

df = tcx_to_df(filepath)
# get pace data from distance column as the difference between the distance of two successive timestamps divided by the difference between the timestamps

df['Distance'] = df['Distance'].astype(float)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

dx = df['Distance'].diff()
dt = (df['Timestamp'].diff().dt.total_seconds())

df['pace'] = (dt / dx) / 60 * 1000     # this is minutes per km

# smoothen the pace column by taking a roughly 10s (10 datapoints) avg
df['pace'] = df['pace'].rolling(10).mean()