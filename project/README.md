# Data Engineering Zoomcamp Final Project: ETL Pipeline for Pharmaceutical Production Data Analysis

## Introduction
This project uses the techniques learned throughout the de zoomcamp course to extract, transform, and load data from the large-scale manufacturing of Insulin into a dashboard for analysis. 

## Background
Insulin is a vital protein product used in the treatment of diabetes. It is produced at scale using sophisticated biopharmaceutical equipment to grow cells which produce the Insulin protein and purify it from impurities. To understand the quantity of Insulin produced, samples are typically taken at several intervals throughout the manufacturing process. This sampling is invasive and leads to delays in manufacturing and introduces risks of contamination. Raman spectroscopy is a powerful analytical technique that is becoming popular in biopharmaceutical production to analyze the composition of biological components as it is non-invasive and measurements can be taken in process eliminating the need for sampling. This technique relies on the spectrum of light scattered through a solution to measure protein concentration and quality attributes. 

The raw data used in this project includes 100 batches worth of processing data for the large-scale manufacutring of Insulin. Each batch includes many records of data from both sample measurements as well as Raman sepctra readings throughout the manufacturing of the Insulin product (for more info on the data set used please refer to [Data Sources](#data-sources) and [Acknowledgements](#acknowledgements)).

## Project Description
The goal of this project is to calculate the Penicillin concentration using only the Raman measurement data for each record of the data and compare the results to the actual measured concentration of Penicillin product (from the Penicillin concentration(P:g/L) column). The first 50 batches of the dataset will be used to train and test the model and the final 50 batches will be ingested to the cloud one record at a time in batch using airflow to simulate near real time ingestion of the manufacturing data. As each record is inserted, the model will calculate the Penicillin concentration using the Raman measurement data and add the calculated Penicillin concentration result to the dataset. The final feature data will be used to create a dashboard analyzing accuracy of the model as well as the distribution of Insulin concentration between both sample measurements and model calculation results.

To achieve this, this project will focus specifically on the following columns from the 100_Batches_IndPenSim_V3.csv dataset:
- Time (h): time attribute measurements were taken (numeric)
- Penicillin concentration(P:g/L): measured concentration of Penicillin product sample (numeric)
- Batch ID: unique identifier of batch of Penicillin produced (integer)
- Fault Flag: indifier for any issues during Raman spec measurement (integer)
- {350:1750}: list of columns with Raman measurement data where the number corresponds to the wavelength in nm of light used for the measurement (numeric)

## Project Structure
The project is organized into the following directories:

1. `src`: Contains source code for the ETL pipeline.
2. `data`: Stores raw and processed data files.
3. `model`: Contains all code related to model development (adapted from code originally created by Shashank Gupta, Ricardo Flores, and Rakesh Bobbala - see [Acknowledgements](#acknowledgements) for model development)

## Requirements
- Python 3.11
- SQL Database (e.g., PostgreSQL, MySQL)
- Pandas library for data manipulation
- SQLAlchemy library for database interactions

## Installation
1. Clone this repository:
```
git clone https://github.com/yourusername/data-engineering-project.git
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

## Usage
1. Configure database connection details in `config.py`.
2. Run the ETL pipeline:
```
python src/etl_pipeline.py
```

## Data Sources
- **Sales Database**: Raw sales data stored in a SQL database.
- **CSV Files**: Additional sales data stored in CSV files.

## ETL Process
1. **Extraction**: Retrieve raw data from the sales database and CSV files.
2. **Transformation**: Clean, standardize, and transform the data into a consistent format.
3. **Loading**: Load the transformed data into a target database for analysis.

## Database Schema
The database schema includes tables for storing transformed sales data, such as `sales`, `customers`, and `products`.

## Author
- Jesse Delzio <jmdelzio@ucdavis.edu>

## License
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgements
- This project was inspired by [2024 Data Engineering Zoomcamp](https://datatalks.club/blog/data-engineering-zoomcamp.html) offered by DataTalks.Club.
- The data used for this project was provided by [kaggle](https://www.kaggle.com/datasets/stephengoldie/big-databiopharmaceutical-manufacturing)
- The model development for this project was inspired by [Shashank Gupta, Ricardo Flores, and Rakesh Bobbala](https://www.kaggle.com/code/wrecked22/regression-analysis)