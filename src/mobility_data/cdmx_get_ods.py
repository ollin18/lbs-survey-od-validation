import geopandas as gpd
import pandas as pd
import requests
import os
from pathlib import Path
import zipfile
import io
import requests
import numpy as np


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

from create_ods import *

#  spark = SparkSession.builder \
#      .appName("MobilityAnalysis") \
#      .config("spark.driver.memory", "96g") \
#      .config("spark.driver.maxResultSize", "90g") \
#      .config("spark.sql.adaptive.enabled", "true") \
#      .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
#      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
#      .getOrCreate()

get_od(area = "cdmx",
        output_dir="../../data/intermediate/od_pairs/",
        geom_dir="../../data/intermediate/geometries/",
        geom_columns=["geomid", "geometry", "population"],
        mobility_dir="/data/Berkeley/",
        spark = spark)

#  spark.stop()
