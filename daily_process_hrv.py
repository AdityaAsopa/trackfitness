import traceback
from pathlib import Path
import os
import json
import glob
import datetime
import orthostatic_delta_hrv_test as hrv
import morning_cardiopulmonary_check as mcp

# Path to the folder you want to check
with open('env.json') as f:
    env = json.load(f)

folder_path = env['ecgdata_folder_path']

# Get a list of all files in the folder which match the pattern: _RR.txt
files = glob.glob(os.path.join(folder_path, "Polar_H10_AEFD6B29_2025*.txt")) #Polar_H10_AEFD6B29_20250209_072031_RR

print('Total files:', len(files))

# def run_batch():
print(files)
failed_files = []
for file in files:
    file = Path(file)
    print(f'Processing file {file}')
    try:
        # hrv.main(file, 'Polar Sensor Logger Export', pre_baseline_period=180, post_baseline_period=60, plot=True)
        mcp.main(file, origin='Polar Sensor Logger Export')
        # _ = os.system('cls')
    except Exception as e:
        print('X' * 80)
        print(f'Error processing file {file}: {e}')
        failed_files.append(Path(file).stem)    

print('ALL DONE!')        
if failed_files:
    print('total failed files:', len(failed_files))
    print(f'Failed files: {failed_files}')
