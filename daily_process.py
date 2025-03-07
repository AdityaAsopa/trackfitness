import os
import json
import glob
import datetime
import orthostatic_delta_hrv_test as hrv

# Path to the folder you want to check
with open('env.json') as f:
    env = json.load(f)

folder_path = env['ecgdata_folder_path']

# Get a list of all files in the folder which match the pattern: _RR.txt
files = glob.glob(os.path.join(folder_path, "Polar_H10_*_RR.txt"))

for file in files:
    # Get the time the file was last modified
    file_time = os.path.getmtime(file)

    # If the file was modified today, process it
    # if datetime.date.fromtimestamp(file_time) == datetime.date.today():
    # Replace this with the code to process the file
    print(f'Processing file {file}')
    failed_files = []
    try:
        hrv.main(file, 'Polar Sensor Logger Export', pre_baseline_period=180, post_baseline_period=120, plot=True)
    except Exception as e:
        print('X' * 80)
        print(f'Error processing file {file}: {e}')
        failed_files.append(file)

print('ALL DONE!')        
if failed_files:
    print(f'Failed files: {failed_files}')