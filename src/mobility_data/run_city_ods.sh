#!/usr/bin/env bash
python city_get_ods.py
python city_get_ods.py --output-file cdmx_agebs.csv --geom-file cdmx_agebs_zm.geojson
python city_get_ods.py --area guadalajara --output-file guadalajara_agebs.csv --geom-file guadalajara_agebs_zm.geojson
