# Mobility Data - OD Pair Generation

This package processes mobility data to generate Origin-Destination (OD) pairs for spatial analysis.

## Files

- `create_ods.py`: Core module containing the `get_od()` function and Spark session configuration
- `cdmx_get_ods.py`: Command-line script to generate OD pairs for CDMX or other areas

## Requirements

- Python 3.x
- PySpark
- geopandas
- pandas
- shapely
- numpy

## Usage

### Basic Usage (Default Parameters)

Run with default parameters for CDMX:

```bash
python cdmx_get_ods.py
```

This will:
- Process mobility data for CDMX
- Read geometries from `../../data/intermediate/geometries/`
- Read mobility data from `/data/Berkeley/`
- Save output to `../../data/intermediate/od_pairs/cdmx_od_geomid.csv`

### Custom Parameters

You can override any parameter using command-line arguments:

```bash
python cdmx_get_ods.py --area guadalajara --mobility-dir /path/to/mobility/data
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--area` | string | `cdmx` | Area name (e.g., cdmx, guadalajara) |
| `--output-dir` | string | `../../data/intermediate/od_pairs/` | Output directory for OD pairs |
| `--output-file` | string | `None` | Custom output filename (default: {area}_od_geomid.csv) |
| `--geom-dir` | string | `../../data/intermediate/geometries/` | Directory containing geometry files |
| `--geom-file` | string | `None` | Geometry filename within geom-dir (default: {area}_geometries.geojson) |
| `--geom-columns` | list | `geomid geometry population` | Columns to read from geometry file |
| `--mobility-dir` | string | `/data/Berkeley/` | Base directory containing mobility data |

### Examples

1. Process Guadalajara data:
```bash
python cdmx_get_ods.py --area guadalajara
```

2. Custom output location:
```bash
python cdmx_get_ods.py --output-dir /path/to/output --output-file custom_od.csv
```

3. Use a custom geometry file:
```bash
python cdmx_get_ods.py --geom-file custom_geometries.geojson
```

4. Different geometry columns:
```bash
python cdmx_get_ods.py --geom-columns id geometry pop density
```

5. Full custom configuration:
```bash
python cdmx_get_ods.py \
  --area cdmx \
  --output-dir ./output/ \
  --geom-dir ./geometries/ \
  --mobility-dir /data/mobility/ \
  --geom-columns geomid geometry population
```

### Help

To see all available options:

```bash
python cdmx_get_ods.py --help
```

## How It Works

The script:

1. Loads geometry data for the specified area from a GeoJSON file
2. Reads mobility data from Parquet files using PySpark
3. Filters mobility data to the bounding box of the geometry
4. Performs spatial joins to assign home and work locations to geometry IDs
5. Aggregates OD pairs and counts unique users
6. Adds population data for home and work locations
7. Saves the result as a CSV file

## Spark Configuration

The Spark session is configured in `create_ods.py` with:
- 96GB driver memory
- 90GB max result size
- Adaptive query execution enabled
- Kryo serialization for performance

## Output Format

The output CSV contains:
- `home_geomid`: Geometry ID of home location
- `work_geomid`: Geometry ID of work location
- `count_uid`: Number of unique users for this OD pair
- `home_population`: Population at home geometry
- `work_population`: Population at work geometry
