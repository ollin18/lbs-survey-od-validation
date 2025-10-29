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

spark = SparkSession.builder \
    .appName("MobilityAnalysis") \
    .config("spark.driver.memory", "96g") \
    .config("spark.driver.maxResultSize", "90g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()


def create_spatial_index(gdf):
    """Create a spatial index for faster lookups"""
    spatial_index = {}
    for idx, row in gdf.iterrows():
        bounds = row['geometry'].bounds  # (minx, miny, maxx, maxy)
        geomid = row['geomid']
        spatial_index[geomid] = {
            'geometry': row['geometry'],
            'bounds': bounds
        }
    return spatial_index

def get_od(area = "cdmx",
           output_dir="../../data/intermediate/od_pairs/",
           geom_dir="../../data/intermediate/geometries/",
           geom_columns=["geomid", "geometry", "population"],
           mobility_dir="/data/Berkeley/",
           spark = spark):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    area_dict = {"cdmx": "MX", "guadalajara": "MX"}

    gdf = gpd.read_file(os.path.join(geom_dir, f"{area}_geometries.geojson"),
                        columns=geom_columns)

    bbox = gdf.total_bounds

    mobility_path = os.path.join(mobility_dir, area_dict[area], "large_quadrant", "relaxed")
    mob = spark.read.parquet(os.path.join(mobility_path, "*.parquet"))

    mob = mob.filter(col("home_latitude").isNotNull() & col("work_latitude").isNotNull())
    mob = mob.filter(
        (col("home_latitude") >= bbox[1]) & (col("home_latitude") <= bbox[3]) &
        (col("home_longitude") >= bbox[0]) & (col("home_longitude") <= bbox[2]) &
        (col("work_latitude") >= bbox[1]) & (col("work_latitude") <= bbox[3]) &
        (col("work_longitude") >= bbox[0]) & (col("work_longitude") <= bbox[2])
    )

    mob = mob.dropDuplicates(["uid"])

    spatial_index = create_spatial_index(gdf)
    spatial_index_broadcast = spark.sparkContext.broadcast(spatial_index)

    def find_containing_polygon_optimized(lon, lat):
        if lon is None or lat is None:
            return None

        point = Point(lon, lat)
        spatial_idx = spatial_index_broadcast.value

        # First pass: quick bounding box check
        candidates = []
        for geomid, data in spatial_idx.items():
            bounds = data['bounds']
            if bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]:
                candidates.append((geomid, data['geometry']))

        # Second pass: polygon containment check only for candidates
        for geomid, polygon in candidates:
            try:
                if polygon.contains(point):
                    return geomid
            except:
                continue
        return None

    find_polygon_optimized_udf = udf(
        find_containing_polygon_optimized,
        StringType()
    )

    print("Creating join...")
    mob = mob.withColumn(
        "home_geomid",
        find_polygon_optimized_udf(col("home_longitude"), col("home_latitude"))
    ).withColumn(
        "work_geomid",
        find_polygon_optimized_udf(col("work_longitude"), col("work_latitude"))
    )

    mob = mob.select([
        col("uid"),
        col("home_geomid"),
        col("work_geomid")
    ])

    mob_grouped = mob.groupBy("home_geomid", "work_geomid").agg({"uid": "count"}).withColumnRenamed("count(uid)", "count_uid")

    od_pairs = mob_grouped.toPandas()

    # Add home and work population
    od_pairs = od_pairs.merge(
            gdf[['geomid', 'population']].rename(columns={'geomid':
                                                          'home_geomid',
                                                          'population':
                                                          'home_population'}),
            on='home_geomid', how='left')

    od_pairs = od_pairs.merge(
            gdf[['geomid', 'population']].rename(columns={'geomid':
                                                          'work_geomid',
                                                          'population':
                                                          'work_population'}),
            on='work_geomid', how='left')

    od_pairs = od_pairs.fillna(0)
    od_pairs.loc[:, 'home_population'] = od_pairs['home_population'].astype(int)
    od_pairs.loc[:, 'work_population'] = od_pairs['work_population'].astype(int)

    output_file = output_dir / f"{area}_od_geomid.csv"
    od_pairs.to_csv(output_file, index=False)
    spark.stop()
    print(f"Saved {len(od_pairs)} workers data to {output_file}")
