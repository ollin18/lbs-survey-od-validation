# OD Survey Analysis Tool - Usage Guide

This guide explains how to use the refactored `od_survey_analysis.py` module for analyzing Origin-Destination (OD) data.

## Overview

The `od_survey_analysis.py` module is a modular, reproducible tool that:
- Analyzes OD data from Location-Based Services (LBS)
- Compares LBS data with survey data
- Generates comprehensive visualizations
- Can be easily adapted for different cities

## Quick Start

### Option 1: Command Line (Default CDMX paths)

```bash
python od_survey_analysis.py --city cdmx
```

This uses the default file paths configured for CDMX.

### Option 2: Command Line (Custom paths)

```bash
python od_survey_analysis.py \
    --city "your_city" \
    --od-data "path/to/od_pairs.csv" \
    --geometry "path/to/geometries.geojson" \
    --survey-od "path/to/survey_od_matrix.csv" \
    --output-dir "path/to/output/figures"
```

### Option 3: Programmatic Usage

```python
from od_survey_analysis import AnalysisConfig, run_analysis

# Create configuration
config = AnalysisConfig(
    city_name="guadalajara",
    od_data_path="data/guadalajara/od_pairs.csv",
    geometry_path="data/guadalajara/geometries.geojson",
    survey_od_path="data/guadalajara/survey_od.csv",
    output_dir="figures/guadalajara",
    rounding_factor=5,
    figure_dpi=300
)

# Run analysis
run_analysis(config)
```

## Input Data Requirements

### 1. OD Data CSV (`od_data_path`)
Required columns:
- `home_geomid` (string): Home district ID
- `work_geomid` (string): Work district ID
- `count_uid` (numeric): Number of unique users

Example:
```csv
home_geomid,work_geomid,count_uid
001,002,150
001,003,200
002,001,175
```

### 2. Geometry GeoJSON (`geometry_path`)
Required fields:
- `geomid` (string): District ID
- `population` (numeric): District population
- Geometry information (polygons)

### 3. Survey OD Matrix CSV (`survey_od_path`)
Required columns:
- `home_geomid` (string): Home district ID
- `work_geomid` (string): Work district ID
- `counts` (numeric): Number of trips from survey

Example:
```csv
home_geomid,work_geomid,counts
001,002,5000
001,003,7500
002,001,6200
```

## Output Files

The analysis generates the following figures in the `output_dir`:

1. **Expansion Factors**
   - `district_expansion_factor_distribution.png`: Distribution of expansion factors
   - `district_expansion_factor_map.png`: Choropleth map of expansion factors

2. **Population and Users**
   - `district_population_map.png`: Population distribution map
   - `district_lbs_uid_map.png`: LBS unique users map
   - `district_scaled_population_scatter.png`: Comparison of scaled users vs population

3. **OD Matrices**
   - `district_lbs_OD_heatmap.png`: LBS OD matrix heatmap
   - `district_survey_OD_heatmap.png`: Survey OD matrix heatmap

4. **Comparison**
   - `district_lbs_survey_correlation_scatter.png`: LBS vs Survey correlation plot

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `city_name` | City name (for titles and paths) | Required |
| `od_data_path` | Path to OD pairs CSV | Required |
| `geometry_path` | Path to geometries GeoJSON | Required |
| `survey_od_path` | Path to survey OD matrix CSV | Required |
| `output_dir` | Output directory for figures | Required |
| `rounding_factor` | Factor for rounding expansion factors | 5 |
| `figure_dpi` | DPI for saved figures | 300 |

## Adding a New City

To analyze a new city, you have two options:

### Option A: Use Command Line

```bash
python od_survey_analysis.py \
    --city "monterrey" \
    --od-data "../../data/intermediate/od_pairs/monterrey_od_geomid.csv" \
    --geometry "../../data/intermediate/geometries/monterrey_geometries.geojson" \
    --survey-od "../../data/clean/monterrey/survey/od_matrix.csv" \
    --output-dir "../../figures/monterrey"
```

### Option B: Add a Configuration Function

Edit `od_survey_analysis.py` and add a function like this:

```python
def create_config_for_monterrey() -> AnalysisConfig:
    """Create configuration for Monterrey analysis."""
    return AnalysisConfig(
        city_name="monterrey",
        od_data_path="../../data/intermediate/od_pairs/monterrey_od_geomid.csv",
        geometry_path="../../data/intermediate/geometries/monterrey_geometries.geojson",
        survey_od_path="../../data/clean/monterrey/survey/od_matrix.csv",
        output_dir="../../figures/monterrey",
        rounding_factor=5,
        figure_dpi=300
    )
```

Then update the `main()` function to recognize "monterrey" as a city with defaults.

## Module Structure

The analysis pipeline consists of these steps:

1. **Data Loading** (`load_data`)
   - Loads OD pairs and geometries

2. **Expansion Factor Calculation** (`calculate_expansion_factors`)
   - Computes expansion factors: population / unique_users
   - Calculates scaled user counts

3. **Visualizations** (multiple functions)
   - Distribution plots
   - Choropleth maps
   - Scatter plots

4. **OD Matrix Creation** (`create_od_matrix`)
   - Aggregates OD pairs with expansion factors
   - Creates wide-format matrix

5. **Survey Data Loading** (`load_survey_od_matrix`)
   - Loads and formats survey OD matrix

6. **Comparison Analysis** (`compare_od_matrices`)
   - Computes correlations (overall, intra-district, inter-district)
   - Creates comparison visualizations

## Example Workflow

```python
# 1. Import the module
from od_survey_analysis import AnalysisConfig, run_analysis

# 2. Set up configuration for your city
config = AnalysisConfig(
    city_name="my_city",
    od_data_path="data/my_city_od.csv",
    geometry_path="data/my_city_geo.geojson",
    survey_od_path="data/my_city_survey.csv",
    output_dir="figures/my_city"
)

# 3. Run the complete analysis
run_analysis(config)

# Or run individual steps for custom analysis:
from od_survey_analysis import load_data, calculate_expansion_factors, plot_maps

df, gdf = load_data(config)
population = calculate_expansion_factors(df, gdf)
plot_maps(population, config)
```

