import geopandas as gpd
import pandas as pd
import requests
import os
from pathlib import Path
import zipfile
import io
import requests
import numpy as np
import argparse


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

from create_ods import *


def main():
    parser = argparse.ArgumentParser(
        description='Generate Origin-Destination (OD) pairs from mobility data for CDMX or other areas.'
    )

    parser.add_argument(
        '--area',
        type=str,
        default='cdmx',
        help='Area name (default: cdmx)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../data/intermediate/od_pairs/',
        help='Output directory for OD pairs (default: ../../data/intermediate/od_pairs/)'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output filename (default: {area}_od_geomid.csv)'
    )

    parser.add_argument(
        '--geom-dir',
        type=str,
        default='../../data/intermediate/geometries/',
        help='Directory containing geometry files (default: ../../data/intermediate/geometries/)'
    )

    parser.add_argument(
        '--geom-file',
        type=str,
        default=None,
        help='Geometry filename within geom-dir (default: {area}_geometries.geojson)'
    )

    parser.add_argument(
        '--geom-columns',
        nargs='+',
        default=['geomid', 'geometry', 'population'],
        help='Columns to read from geometry file (default: geomid geometry population)'
    )

    parser.add_argument(
        '--mobility-dir',
        type=str,
        default='/data/Berkeley/',
        help='Base directory containing mobility data (default: /data/Berkeley/)'
    )

    args = parser.parse_args()

    get_od(
        area=args.area,
        output_dir=args.output_dir,
        output_file=args.output_file,
        geom_dir=args.geom_dir,
        geom_file=args.geom_file,
        geom_columns=args.geom_columns,
        mobility_dir=args.mobility_dir,
        spark=spark
    )


if __name__ == '__main__':
    main()

