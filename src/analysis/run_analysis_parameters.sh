#!/usr/bin/env bash
python od_survey_analysis.py \
    --city guadalajara \ # name in the plots
    --od-data "../../data/intermediate/od_pairs/guadalajara_agebs.csv"\
    --geometry "../../data/intermediate/geometries/guadalajara_agebs_zm.geojson"\
    --survey-od None \
    --output-dir "../../figures/guadalajara"

python od_survey_analysis.py --city guadalajara_od --od-data "../../data/intermediate/od_pairs/guadalajara_agebs.csv" --geometry "../../data/intermediate/geometries/guadalajara_geometries.geojson" --survey-od "../../data/intermediate/od_pairs/guadalajara_od_geomid.csv" --output-dir "../../figures/guadalajara_od"
