import pandas as pd
import geopandas as gpd
import os

geom_path = '../../data/raw/cdmx_geoms/DistritosEODHogaresZMVM2017.shp'

# Load the shapefile
gdf = gpd.read_file(geom_path, encoding='latin1')
gdf = gdf.to_crs(epsg=4326)
gdf.rename(columns={'Distrito': 'geomid'}, inplace=True)
gdf = gdf[['geomid', 'geometry']]

# Get population data from survey
tvivienda = pd.read_csv("../../data/raw/cdmx_survey/tvivienda_eod2017/conjunto_de_datos/tvivienda.csv")
tvivienda["population"] = tvivienda["p1_1"] * tvivienda["factor"]
tvivienda["distrito"] = tvivienda["distrito"].astype(str).str.zfill(3)

population = tvivienda.groupby("distrito")["population"].sum().reset_index()
population.rename(columns={"distrito": "geomid"}, inplace=True)

gdf = gdf.merge(population, on='geomid', how='left')
gdf['population'] = gdf['population'].fillna(0).astype(int)

os.makedirs(os.path.dirname('../../data/intermediate/geometries/'),
            exist_ok=True)
output_path = '../../data/intermediate/geometries/cdmx_geometries.geojson'
gdf.to_file(output_path, driver='GeoJSON')
