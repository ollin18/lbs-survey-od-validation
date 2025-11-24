# Departure Time Analysis

A modular, reproducible framework for analyzing departure times from location-based services (LBS) data.

## Overview

This analysis tool processes trip data extracted from mobile location data to characterize departure time patterns and validate them against survey data. It supports:
- **Multiple trip types**: Home-Work (HW), Home-Other (HO), Non-Home (NonH), All trips
- **Sampling strategies**: Informed (distribution-based), Random uniform, Raw data
- **Reusable computations**: Caching of processed trips for fast subsequent analyses
- **Batch execution**: Run single or multiple configurations in sequence
- **Comparison analysis**: Compare predicted departure times with survey distributions using Dynamic Time Warping (DTW)

## Structure

```
departure_time/
├── config_template.py          # Configuration definitions for all analyses
├── departure_time_analysis.py  # Core analysis functions
├── run_analysis.py             # Runner script for executing analyses
└── README.md                   # This file
```

## Quick Start

### 1. List available configurations

```bash
python run_analysis.py --list
```

### 2. Run a single analysis

```bash
python run_analysis.py --config HW_informed
```

### 3. Run multiple analyses

```bash
# Compare sampling strategies
python run_analysis.py --config HW_informed HW_random_uniform HW_raw

# Compare trip types
python run_analysis.py --config HW_informed HO_informed nonH_informed all_informed
```

### 4. Run all analyses

```bash
python run_analysis.py --all
```

### 5. Force recomputation (don't use cached trips)

```bash
python run_analysis.py --config HW_informed --recompute
```

## Available Configurations

### By Trip Type

| Configuration | Description | Sampling |
|--------------|-------------|----------|
| `HW_informed` | Home-to-Work trips | Informed (survey distribution) |
| `HO_informed` | Home-to-Other trips | Informed |
| `nonH_informed` | Non-Home trips | Informed |
| `all_informed` | All trips | Informed |

### By Sampling Strategy

| Configuration | Description | Trip Type |
|--------------|-------------|-----------|
| `HW_informed` | Informed sampling | Home-to-Work |
| `HW_random_uniform` | Random uniform sampling | Home-to-Work |
| `HW_raw` | No sampling (raw data) | Home-to-Work |

## Configuration Details

### Trip Filters

Each configuration can specify:
- **Origin types**: Location types for trip origins (e.g., 'H' for home, 'W' for work)
- **Destination types**: Location types for destinations
- **Weekday only**: Filter to weekdays only
- **Same day**: Only include trips that start and end on the same day
- **First/last selection**: Select first H→W or last W→H trip of the day

### Sampling Strategies

1. **Informed**: Samples departure times based on an observed hourly distribution from survey data
2. **Random Uniform**: Samples uniformly between the end of origin stop and start of destination stop
3. **Raw**: Uses the actual end time of the origin stop (no sampling)

## Programmatic Usage

You can also use the modules directly in your Python code:

```python
from config_template import get_config
from departure_time_analysis import run_analysis

# Get a predefined configuration
config = get_config("HW_informed", country="MX")

# Run the analysis
results = run_analysis(config, recompute_trips=False)

# Results is a pandas DataFrame with hourly distribution
print(results)
```

### Creating Custom Configurations

```python
from config_template import AnalysisConfig, TripFilterConfig, SamplingConfig

config = AnalysisConfig(
    name="custom_analysis",
    country="US",
    parquet_dir_in="/path/to/stops_data",
    temp_dir="/path/to/temp",
    output_dir="/path/to/output",
    trip_filter=TripFilterConfig(
        origin_types=['H'],
        dest_types=['W'],
        weekday_only=True
    ),
    sampling=SamplingConfig(
        strategy="informed",
        distribution_path="/path/to/distribution.csv"
    )
)

results = run_analysis(config)
```

## Output

Each analysis produces:

1. **Processed trips** (cached in `temp_dir`):
   - Parquet files with trip-level data
   - Reused across runs unless `--recompute` is specified

2. **Plots** (saved to `output_dir`):
   - `trips_by_hour_{country}_2023.png`: Single distribution plot
   - `trips_by_hour_comparison_{country}_2023.png`: Comparison with survey data (if available)

3. **CSV results**:
   - `hourly_distribution_{config_name}.csv`: Hourly trip counts and percentages

## Reusing Computations

The framework caches processed trips in the `temp_dir` to avoid recomputation:

- **First run**: Processes all trips from raw data (slow)
- **Subsequent runs**: Reads from cached parquet files (fast)
- **Force recompute**: Use `--recompute` flag to regenerate trips

This allows you to:
1. Process trips once with a specific trip filter and sampling strategy
2. Generate multiple plots or analyses from the same processed trips
3. Compare different analyses without reprocessing the raw data

## Spark Configuration

Default Spark settings are defined in `SparkConfig`:
- 20 cores on local mode
- 60GB driver memory
- Optimized for single-node processing

You can modify these in `config_template.py` or programmatically:

```python
from config_template import SparkConfig

custom_spark = SparkConfig(
    master="local[40]",
    driver_memory="120g",
    shuffle_partitions=80
)

config.spark_config = custom_spark
```

## Comparison with Survey Data

If a survey distribution file is provided, the analysis will:
1. Load the survey hourly distribution
2. Compute Dynamic Time Warping (DTW) distance between LBS and survey distributions
3. Generate comparison plots with correlation metrics

Survey distribution CSV should have columns:
- `hour`: Hour of day (0-23)
- `percentage`: Percentage of trips in that hour

## Examples

### Compare all Home-Work sampling methods

```bash
python run_analysis.py --config HW_informed HW_random_uniform HW_raw
```

This will generate three sets of plots showing how different sampling strategies affect the departure time distribution.

### Analyze all trip types with informed sampling

```bash
python run_analysis.py --config HW_informed HO_informed nonH_informed all_informed
```

This compares departure time patterns for different trip purposes.

### Process a different country

```bash
python run_analysis.py --config HW_informed --country US
```

Note: Make sure the data paths in the configuration match your data structure.

## Troubleshooting

### Memory issues

If you encounter memory errors:
1. Reduce `month_range` in the configuration (process fewer months)
2. Increase Spark memory settings in `SparkConfig`
3. Reduce `repartition_size` in the configuration

### Missing data

If analyses fail with missing data errors:
1. Check that `parquet_dir_in` points to valid stop data
2. Verify survey distribution file exists (if using informed sampling)
3. Ensure required columns exist in the input data

### Slow performance

First runs are slow because they process all trips. To speed up:
1. Process fewer months initially (`month_range=(0, 1)`)
2. Use cached trips for subsequent analyses (don't use `--recompute`)
3. Adjust `repartition_size` based on your data size

## How It Works

### Data Processing Pipeline

1. **Trip Extraction**: Raw stop data (parquet files) is processed to identify individual trips
   - Stops are ordered by timestamp
   - Sequential stops from the same person form a trip
   - Trip origin and destination are determined by stop location types (H=Home, W=Work, O=Other)

2. **Trip Filtering**: Trips are filtered based on configuration:
   - **Origin/destination type**: Filter by location type (e.g., only H→W trips)
   - **Weekday only**: Exclude weekend trips
   - **Same day only**: Exclude trips spanning multiple days
   - **First/last selection**: For HW trips, optionally select only first H→W or last W→H per day
   - **Exclude other**: Remove trips with ambiguous location types

3. **Departure Time Sampling**: The actual departure time is determined based on sampling strategy:
   - **Raw**: Uses the actual end time of the origin stop
   - **Random Uniform**: Samples uniformly between origin stop end and destination stop start
   - **Informed**: Samples from an observed hourly distribution (from survey data)
     - Maps hourly probabilities to cumulative distribution
     - Generates departure times matching the survey's temporal patterns

4. **Hourly Aggregation**: Sampled departure times are binned into 24 hourly buckets
   - Counts trips per hour
   - Calculates percentages
   - Generates visualizations

5. **Validation**: Results are compared with survey data using:
   - **Visual comparison**: Overlaid histograms with correlation metrics
   - **DTW distance**: Dynamic Time Warping distance measures temporal similarity

### Sampling Strategies Explained

**Informed (Distribution-based)**
- Uses actual survey data to specify the probability of departing in each hour
- Best for validating that your synthetic trips match observed behavior
- Requires a survey distribution file with hourly percentages

**Random Uniform**
- Treats all departure times as equally likely between origin stop end and destination stop start
- Provides a baseline that assumes no temporal preferences
- Useful for comparison to understand how much survey patterns matter

**Raw**
- Uses the actual end time of the origin stop
- Reflects only the observed stop timing without additional sampling
- Most direct but potentially biased by incomplete or noisy location data

## Migration from Old Scripts

The old scripts (`departure_time-*.py`) are now replaced by this modular framework:

| Old Script | New Configuration |
|-----------|-------------------|
| `departure_time-HW-informed.py` | `HW_informed` |
| `departure_time-HW-random-uniform.py` | `HW_random_uniform` |
| `departure_time-HW-raw.py` | `HW_raw` |
| `departure_time-HO-informed.py` | `HO_informed` |
| `departure_time-nonH-informed.py` | `nonH_informed` |
| `departure_time-all-informed.py` | `all_informed` |

To run the equivalent analysis:
```bash
# Old way
python departure_time-HW-informed.py

# New way
python run_analysis.py --config HW_informed
```
