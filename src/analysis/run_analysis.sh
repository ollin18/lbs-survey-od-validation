#!/usr/bin/env bash
python od_survey_analysis.py \
    --city monterrey \ # name in the plots
    --od-data "path/to/monterrey_od.csv" \ # origin-destination data
    --geometry "path/to/monterrey_geo.geojson" \ # geographic boundaries
    --survey-od "path/to/monterrey_survey.csv" \ # survey data if available
    --output-dir "figures/monterrey" # output directory for figures

python od_survey_analysis.py \
    --city guadalajara \ # name in the plots
    --od-data "../../data/intermediate/od_pairs/guadalajara_agebs.csv"\
    --geometry "../../data/intermediate/geometries/guadalajara_agebs_zm.geojson"\
    --survey-od None \
    --output-dir "../../figures/guadalajara"
