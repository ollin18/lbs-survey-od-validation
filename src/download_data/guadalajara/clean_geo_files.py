import geopandas as gpd
import pandas as pd
import os

# NOTE: All files are downloaded from https://zoom.imeplan.mx/mapa manually
# There is no programmatic way to download them

#  Geoms
data_dir = os.path.join("..", "..", "..", "data")
raw_dir = os.path.join(data_dir, "raw", "guadalajara", "geoms")
gdf = gpd.read_file(os.path.join(raw_dir, "zonificacion_eod", "amg_origenes_y_destinos_eodPolygon.shp"))

# Population

pop = gpd.read_file(os.path.join(raw_dir, "habitantes_vivienda", "amg_habitantes_por_vivPolygon.shp"))

# Dictionary names
names_dict = {
    "numero_de_": "geomid",
    "nombre_de_": "geom_name",
    "habitantes": "population"
}

gdf.rename(columns=names_dict, inplace=True)
pop.rename(columns=names_dict, inplace=True)

gdf = gdf[["geomid", "geom_name", "geometry"]]
pop = pop[["geomid", "geom_name", "population", "geometry"]]

intermediate_dir = os.path.join(data_dir, "intermediate", "geometries")
pop.to_file(os.path.join(intermediate_dir, "guadalajara_geometries.geojson"),
            driver="GeoJSON")
