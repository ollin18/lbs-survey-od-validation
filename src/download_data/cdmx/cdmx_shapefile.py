import pandas as pd
import zipfile
import requests
import os

data_url = "https://giitral.iingen.unam.mx/Estudios/assets/img/mapas/hogares/DistritosEODHogaresZMVM2017.zip"

os.makedirs('../../data/raw/cdmx_survey/', exist_ok=True)

response = requests.get(data_url)
zip_path = '../../data/raw/cdmx_survey/eod_zmvm_2017_shp.zip'
with open(zip_path, 'wb') as f:
    f.write(response.content)

extract_path = '../../data/raw/cdmx_geoms/'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
