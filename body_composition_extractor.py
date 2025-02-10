import json
import re
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from typing import Union

# Configure Tesseract executable path
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class BodyCompositionExtractor:
    """
    A class for extracting numerical body composition data from images.
    
    The extractor crops predefined regions from the image, applies OCR, and then
    parses the numerical value from the result.
    """
    # Coordinates defined as (x, y, width, height) for each measurement.
    COORDINATES = {
        'Body Weight': (60, 700, 150, 60),
        'Body Fat': (60, 1054, 150, 60),
        'Body Water': (60, 1408, 150, 60),
        'BMR': (60, 1762, 200, 60),
        'Visceral Fat': (60, 2116, 127, 60),
        'Protein Mass': (60, 2470, 140, 60),
        'BMI': (600, 700, 150, 60),
        'Muscle Rate': (600, 1054, 156, 60),
        'Bone Mass': (600, 1408, 122, 60),
        'Metabolic Age': (600, 1762, 90, 60),
        'Subcutaneous Fat': (600, 2116, 155, 60),
        'Muscle Mass': (600, 2470, 153, 60)
    }

    def __init__(self):
        pass

    def extract_numbers_from_image(self, image_path: Union[str, Path]) -> pd.DataFrame:
        """
        Extract numerical values from a body composition image.
        
        Args:
            image_path (str or Path): The path to the image file.
            
        Returns:
            pd.DataFrame: A single-row dataframe containing the extracted values.
        """
        image_path = Path(image_path)
        try:
            img = Image.open(image_path)
            # Optionally convert to grayscale to speed up OCR:
            img = img.convert('L')
        except Exception as e:
            raise RuntimeError(f"Unable to open image {image_path}: {e}")

        numbers = {}
        for label, (x, y, w, h) in self.COORDINATES.items():
            # Crop the image for the specified measurement area.
            cropped = img.crop((x, y, x + w, y + h))
            ocr_config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.%'
            ocr_result = pytesseract.image_to_string(cropped, lang='eng', config=ocr_config)
            value = self._parse_numeric_value(ocr_result)
            numbers[label] = value

        # Return the numbers as a one-row DataFrame.
        return pd.DataFrame([numbers])

    @staticmethod
    def _parse_numeric_value(text: str) -> float:
        """
        Parse the first occurrence of a numeric value from text using regex.
        Returns np.nan if no valid number is found.
        
        Args:
            text (str): The OCR output text.
            
        Returns:
            float: The extracted number or np.nan.
        """
        match = re.search(r'[\d.]+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return np.nan
        return np.nan

    @staticmethod
    def parse_datetime_from_filename(filepath: Union[str, Path]) -> datetime.datetime:
        """
        Extract a datetime from the filename.
        
        The expected filename format is 'DrTrust-mm-dd-yyyy_hh_mm_ss.jpg'.
        
        Args:
            filepath (str or Path): The full path of the image file.
            
        Returns:
            datetime.datetime: The parsed datetime object.
        """
        filepath = Path(filepath)
        # Remove the 'DrTrust-' prefix; the datetime part starts at character index 8.
        datetime_str = filepath.stem[8:]
        return datetime.datetime.strptime(datetime_str, '%m-%d-%Y_%H_%M_%S')

    @staticmethod
    def update_csv_record(record_path: Union[str, Path], new_data: pd.DataFrame) -> None:
        """
        Append new data to an existing CSV record file (or create it if it doesn't exist).
        
        Args:
            record_path (str or Path): Path to the CSV file.
            new_data (pd.DataFrame): DataFrame containing the new record(s).
        """
        record_path = Path(record_path)
        if record_path.exists():
            try:
                existing_data = pd.read_csv(record_path)
            except Exception as e:
                print(f"Warning: Unable to read existing CSV file, starting a new record. Error: {e}")
                existing_data = pd.DataFrame()
        else:
            existing_data = pd.DataFrame()

        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(record_path, index=False)


def load_config(config_file: Union[str, Path] = 'env.json') -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file (str or Path): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(Path(config_file)) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_file}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_filepath>")
        sys.exit(1)

    image_path = sys.argv[1]
    config = load_config()  # Loads env.json from the current directory
    record_csv_path = config.get('bodycomp_record_folder_path', 'bodycomp_record.csv')

    extractor = BodyCompositionExtractor()
    # Extract numbers from the image.
    data_df = extractor.extract_numbers_from_image(image_path)
    # Parse the date and time from the filename.
    dt = extractor.parse_datetime_from_filename(image_path)
    data_df['date'] = dt.date()
    data_df['time'] = dt.time()

    # Append the new record to the CSV file.
    extractor.update_csv_record(record_csv_path, data_df)
    print("Data extraction and CSV update complete.")


if __name__ == '__main__':
    main()
