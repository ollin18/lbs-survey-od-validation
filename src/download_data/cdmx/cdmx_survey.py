import pandas as pd
import zipfile
import requests
import os

data_url = "https://www.inegi.org.mx/contenidos/programas/eod/2017/datosabiertos/eod_2017_csv.zip"
os.makedirs('../../data/raw/cdmx_survey/', exist_ok=True)

response = requests.get(data_url)
zip_path = '../../data/raw/cdmx_survey/eod_2017_csv.zip'
with open(zip_path, 'wb') as f:
    f.write(response.content)

extract_path = '../../data/raw/cdmx_survey/'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load the dataset
data_file = extract_path + 'eod_2017_cdmx.csv'
df = pd.read_csv(data_file, encoding='latin1')
print(df.head())
