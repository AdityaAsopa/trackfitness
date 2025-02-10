# **TrackFitness**
*A collection of scripts and tools to parse, analyze, and visualize body fitness data from multiple sources.*

## **Overview**
TrackFitness is a set of Python scripts designed to extract, process, and analyze fitness-related data from various sources. This repository includes:

- **Body composition extraction** from image-based data.
- **Runalyze scraping** for fetching fitness metrics.
- **Morning cardiopulmonary check** analysis.
- **Heart rate variability (HRV) and orthostatic load assessment**.

## **Features**
- Extract **body composition metrics** (Body Fat, BMI, BMR, etc.) from images.
- Scrape **Runalyze fitness data** using Selenium.
- Perform **orthostatic and HRV analysis**.
- Generate **breathing pattern metrics** from HR data.
- Provide **visualizations** for deeper insights.

---

## **Installation**
To use this project, install the required dependencies:

```sh
pip install numpy pandas matplotlib seaborn pillow pytesseract selenium beautifulsoup4 scipy
```

Ensure you have:
- **Tesseract OCR** installed and configured for image processing.
- **Selenium WebDriver** installed for web scraping.

---

## **Scripts Overview**
### **1. Body Composition Extractor**
Extracts numerical body composition data from images using OCR.

#### **Usage**
```sh
python body_composition_extractor.py <image_filepath>
```

#### **How It Works**
1. Uses **Tesseract OCR** to extract numerical values from predefined coordinates in an image.
2. Parses values for **Body Weight, Body Fat, BMI, BMR, etc.**.
3. Saves extracted data into a CSV file.

#### **Example**
```python
from body_composition_extractor import BodyCompositionExtractor

extractor = BodyCompositionExtractor()
data = extractor.extract_numbers_from_image("sample_image.jpg")
print(data)
```

---

### **2. Runalyze Scraper**
Scrapes fitness data from **Runalyze** and converts it into a structured dataset.

#### **Usage**
```sh
python runalyze_scrapper.py
```

#### **How It Works**
1. Logs into **Runalyze** using Selenium.
2. Scrapes the **fitness table** and extracts key metrics.
3. Cleans and structures the data into a Pandas DataFrame.
4. Saves data to a CSV file.

#### **Example**
```python
from runalyze_scrapper import RunalyzeScraper

scraper = RunalyzeScraper(
    login_url="https://runalyze.com/login",
    data_url="https://runalyze.com/dashboard",
    username="your_email",
    password="your_password",
    driver_path="path/to/chromedriver"
)

page_source = scraper.get_page_source()
df = scraper.parse_fitness_table(page_source)
print(df)
```

---

### **3. Morning Cardiopulmonary Check**
Analyzes heart rate, breathing, and orthostatic load using HRV data.

#### **Usage**
```sh
python morning_cardiopulmonary_check.py <data_file_path>
```

#### **How It Works**
1. Reads heart rate and RR-interval data from **Polar Sensor Logger** or **Runalyze** exports.
2. Segments the data into **pre, orthostatic, and post-exercise phases**.
3. Calculates:
   - **Breathing rate**
   - **HR recovery rate**
   - **HR variability (HRV) metrics (RMSSD, SDNN)**
   - **Breathing cycle timing**
4. Generates **visual plots**.

#### **Example**
```python
from morning_cardiopulmonary_check import OrthostaticAnalyzer

analyzer = OrthostaticAnalyzer("data.csv")
report = analyzer.generate_report()
print(report)
```

---

## **Walkthrough**
### **1. Capturing Data**
1. **For body composition:** Take an image of your fitness tracker’s body composition report.
2. **For Runalyze scraping:** Ensure your **login credentials** are correctly set.
3. **For HRV analysis:** Use **Polar Sensor Logger** to record HRV data.

### **2. Running the Scripts**
Run the scripts from the terminal or integrate them into your Python projects.

### **3. Viewing Results**
- **CSV files** store extracted and processed data.
- **Graphs and plots** visualize HR trends and breathing patterns.

### **4. Adding Screenshots**
You can insert screenshots of:
- **OCR-extracted data**
- **Runalyze fitness tables**
- **HRV analysis plots**

---

## **Results & Interpretation**
- **HRV Test:** Higher RMSSD indicates better cardiovascular fitness.
- **Breathing Quality:** Balanced inhalation/exhalation cycles suggest good respiratory health.
- **Orthostatic Load:** Significant HR drops after standing indicate proper autonomic function.

---

## **Additional References**
- [Orthostatic Load Wikipedia](https://en.wikipedia.org/wiki/Orthostatic_load)
- [HRV Wikipedia](https://en.wikipedia.org/wiki/Heart_rate_variability)
- [HRV4Training by Marco Altini](https://www.hrv4training.com/)

---

## **Future Improvements**
- **Automate screenshot capture** for OCR verification.
- **Improve OCR accuracy** with better image preprocessing.
- **Enhance HRV insights** with AI-based predictions.

---