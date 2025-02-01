import os
import json
import pytesseract
from PIL import Image
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
import traceback
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_numbers_from_image(image_path):
    """
    Extract numerical values from a single body composition image.
    """
    # Open and process the image
    img = Image.open(image_path)

    # break image into small cropped areas and extract number values from each area
    # area coordinates
    coordinates = {
                    'Body Weight':[60,700,150,60],
                    'Body Fat':[60,1054,150,60],
                    'Body Water':[60,1408,150,60],
                    'BMR':[60,1762,200,60],
                    'Visceral Fat':[60,2116,127,60],
                    'Protein Mass':[60,2470,140,60],
                    'BMI':[600,700,150,60],
                    'Muscle Rate':[600,1054,156,60],
                    'Bone Mass':[600,1408,122,60],
                    'Metabolic Age':[600,1762,90,60],
                    'Subcutaneous Fat':[600,2116,155,60],
                    'Muscle Mass':[600,2470,153,60]
                    }
    # Extract text from each area
    numbers = {}
    fig, ax = plt.subplots(6,2, figsize=(20, 15))
    ax=ax.ravel()
    for k,v in coordinates.items():
        x,y,w,h = v
        area = (x, y, x+w, y+h)
        cropped_img = img.crop(area)
        
        # use tesseract
        text = pytesseract.image_to_string(cropped_img, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.%')

        # make a dict
        text = (text[:-2])
        # convert to numeric value
        print(k, text)
        numbers[k] = float(text)

    # convert dict to dataframe
    print(numbers)
    df = pd.DataFrame(data=numbers, columns=numbers.keys(), index=[0])
    print(df)
    return df

def main():
    import sys
    filepath = sys.argv[1]
    df = extract_numbers_from_image(filepath)
    # extract date from filename
    filename_format = 'DrTrust-mm-dd-yyyy_hh_mm_ss.jpg'
    # get dd-mm-yyyy_hh_mm_ss
    datetimestr = Path(filepath).stem[8:]
    # convert to datetime
    import datetime
    dt = datetime.datetime.strptime(datetimestr, '%m-%d-%Y_%H_%M_%S')
    df['date'] = dt.date()
    df['time'] = dt.time()

    # read in the existing data
    with open('env.json') as f:
        env = json.load(f)

    saved_data_path = env['bodycomp_record_folder_path']
    try:
        existing_data = pd.read_csv(saved_data_path)
    except:
        existing_data = pd.DataFrame()

    # append the new data
    new_data = pd.concat([existing_data, df], axis=0)
    # save as csv
    new_data.to_csv(saved_data_path, index=False)


if __name__ == "__main__":
    main()