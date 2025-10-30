"""
Departure Time Analysis Tool

A modular and reproducible tool for analyzing departure times from location-based services data.
Supports different trip types (HW, HO, nonH, all) and sampling strategies (informed, random, raw).

Usage:
    from departure_time_analysis import run_analysis
    from config_template import get_config

    config = get_config("HW_informed")
    run_analysis(config)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
import os

from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.functions import col, lit, hour, lead, count, row_number
from pyspark.sql.functions import udf
from pyspark.storagelevel import StorageLevel

from config_template import AnalysisConfig, SamplingConfig


# =============================================================================
# SPARK SESSION MANAGEMENT
# =============================================================================

def create_spark_session(config: AnalysisConfig) -> SparkSession:
    """Create and configure Spark session.

    Args:
        config: Analysis configuration

    Returns:
        Configured SparkSession
    """
    # Stop existing session if any
    try:
        spark = SparkSession.getActiveSession()
        if spark:
            spark.stop()
    except:
        pass

    sc = config.spark_config
    spark = (
        SparkSession.builder
        .appName(sc.app_name)
        .master(sc.master)
        .config("spark.driver.memory", sc.driver_memory)
        .config("spark.driver.maxResultSize", sc.driver_max_result_size)
        .config("spark.executor.memory", sc.executor_memory)
        .config("spark.local.dir", sc.local_dir)
        .config("spark.sql.shuffle.partitions", str(sc.shuffle_partitions))
        .config("spark.default.parallelism", str(sc.default_parallelism))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.memory.fraction", str(sc.memory_fraction))
        .config("spark.memory.storageFraction", str(sc.storage_fraction))
        .getOrCreate()
    )

    spark.sparkContext.setCheckpointDir(f"{config.temp_dir}/_chkpt")

    print(f"Spark running on: {spark.sparkContext.master}")
    print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")

    return spark


# =============================================================================
# DISTRIBUTION LOADING AND SAMPLING STRATEGIES
# =============================================================================

def load_hourly_distribution(path: str) -> Dict[int, float]:
    """Load hourly distribution from CSV file.

    Args:
        path: Path to CSV with columns: hour, percentage

    Returns:
        Dictionary mapping hour to percentage
    """
    df = pd.read_csv(path)
    return dict(zip(df['hour'].astype(int), df['percentage']))


def normalize_distribution(dist: Dict[int, float]) -> Dict[int, float]:
    """Normalize distribution to sum to 1.0.

    Args:
        dist: Dictionary mapping hour to percentage/probability

    Returns:
        Normalized distribution
    """
    total = sum(dist.values())
    return {k: v/total for k, v in dist.items()}


def create_sampling_udf(sampling_config: SamplingConfig):
    """Create UDF for sampling departure times based on configuration.

    Args:
        sampling_config: Sampling configuration

    Returns:
        PySpark UDF for sampling departure times
    """
    strategy = sampling_config.strategy

    if strategy == "raw":
        # Raw strategy: use origin end time directly
        def sample_raw(origin_end_epoch, dest_start_epoch):
            return float(origin_end_epoch) if origin_end_epoch is not None else None

        return udf(sample_raw, "double")

    elif strategy == "random_uniform":
        # Random uniform strategy: sample uniformly between origin end and dest start
        def sample_uniform(origin_end_epoch, dest_start_epoch):
            if origin_end_epoch is None or dest_start_epoch is None:
                return None
            return origin_end_epoch + (dest_start_epoch - origin_end_epoch) * np.random.random()

        return udf(sample_uniform, "double")

    elif strategy == "informed":
        # Informed strategy: sample based on hourly distribution
        # Load distribution
        if sampling_config.distribution_path:
            dist = load_hourly_distribution(sampling_config.distribution_path)
        elif sampling_config.distribution_dict:
            dist = sampling_config.distribution_dict
        else:
            raise ValueError("Informed sampling requires distribution_path or distribution_dict")

        dist_norm = normalize_distribution(dist)

        def sample_informed(origin_end_epoch, dest_start_epoch):
            """Sample departure time from distribution within valid window."""
            if origin_end_epoch is None or dest_start_epoch is None:
                return None

            # Convert to datetime
            origin_end = datetime.fromtimestamp(origin_end_epoch)
            dest_start = datetime.fromtimestamp(dest_start_epoch)

            # Get valid hours in the window
            valid_hours = []
            valid_probs = []

            current = origin_end.replace(minute=0, second=0, microsecond=0)
            end = dest_start.replace(minute=59, second=59)

            while current <= end:
                h = current.hour
                if current >= origin_end and current <= dest_start:
                    valid_hours.append(h)
                    valid_probs.append(dist_norm.get(h, 0.01))  # small default prob
                current += timedelta(hours=1)

            if not valid_hours:
                # Fallback to uniform if no valid hours
                return origin_end_epoch + (dest_start_epoch - origin_end_epoch) * np.random.random()

            # Normalize probabilities for valid hours only
            prob_sum = sum(valid_probs)
            if prob_sum > 0:
                valid_probs = [p/prob_sum for p in valid_probs]
            else:
                valid_probs = [1.0/len(valid_hours)] * len(valid_hours)

            # Sample an hour
            sampled_hour = np.random.choice(valid_hours, p=valid_probs)

            # Get timestamp for that hour, sample minute uniformly within the hour
            sampled_dt = origin_end.replace(hour=sampled_hour, minute=0, second=0, microsecond=0)

            # Add random minutes (0-59)
            minute_offset = np.random.randint(0, 60)
            sampled_dt += timedelta(minutes=minute_offset)

            # Ensure it's within bounds
            sampled_epoch = sampled_dt.timestamp()
            if sampled_epoch < origin_end_epoch:
                sampled_epoch = origin_end_epoch
            elif sampled_epoch > dest_start_epoch:
                sampled_epoch = dest_start_epoch

            return sampled_epoch

        return udf(sample_informed, "double")

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


# =============================================================================
# TRIP PROCESSING
# =============================================================================

def process_trips_for_month(
    spark: SparkSession,
    df_clean,
    month: int,
    config: AnalysisConfig
) -> None:
    """Process trips for a single month and save to parquet.

    Args:
        spark: Spark session
        df_clean: Full cleaned dataframe
        month: Month number to process
        config: Analysis configuration
    """
    print(f"\n{'='*60}\nProcessing month {month}...\n{'='*60}")

    trip_filter = config.trip_filter

    # Filter and repartition for this month
    df_month = (
        df_clean
        .filter(col("month") == month)
        .repartition(config.repartition_size)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    print(f"Rows in month {month}: {df_month.count():,}")

    # Window by UID in time order
    w = Window.partitionBy("uid").orderBy("stop_event")

    # Add next-stop information
    df_with_next = (
        df_month
        .withColumn("next_cluster_label", lead("cluster_label").over(w))
        .withColumn("dest_start_epoch", lead(col("start_timestamp")).over(w))
        .withColumn("dest_location_type", lead(col("location_type")).over(w))
        .withColumn("dest_weekend", lead(col("weekend")).over(w))
    )

    # Build filter conditions
    filter_conditions = [
        col("next_cluster_label").isNotNull(),
        col("cluster_label") != col("next_cluster_label"),
        col("end_timestamp").isNotNull(),
        col("dest_start_epoch").isNotNull(),
        col("dest_start_epoch") > col("end_timestamp")
    ]

    # Weekday filter
    if trip_filter.weekday_only:
        filter_conditions.extend([
            col("weekend") == F.lit(False),
            col("dest_weekend") == F.lit(False)
        ])

    # Location type filters
    if trip_filter.origin_types is not None and trip_filter.dest_types is not None:
        # Specific origin and destination (e.g., H->W)
        origin_dest_condition = None
        for origin_type in trip_filter.origin_types:
            for dest_type in trip_filter.dest_types:
                condition = (
                    (col("location_type") == origin_type) &
                    (col("dest_location_type") == dest_type)
                )
                if origin_dest_condition is None:
                    origin_dest_condition = condition
                else:
                    origin_dest_condition = origin_dest_condition | condition

        # Add reverse direction for bidirectional filters (H->W or W->H)
        if 'H' in trip_filter.origin_types and 'W' in trip_filter.dest_types:
            if 'W' in trip_filter.origin_types and 'H' in trip_filter.dest_types:
                # Already covered both directions
                pass
            else:
                # Add W->H as well
                origin_dest_condition = origin_dest_condition | (
                    (col("location_type") == "W") & (col("dest_location_type") == "H")
                )

        filter_conditions.append(origin_dest_condition)

    elif trip_filter.origin_types is not None:
        # Specific origin type only (e.g., H->anywhere except H)
        origin_condition = None
        for origin_type in trip_filter.origin_types:
            condition = (
                (col("location_type") == origin_type) &
                (col("dest_location_type") != origin_type)
            )
            if origin_condition is None:
                origin_condition = condition
            else:
                origin_condition = origin_condition | condition
        filter_conditions.append(origin_condition)

    elif trip_filter.dest_types is not None:
        # Specific destination type only
        dest_condition = None
        for dest_type in trip_filter.dest_types:
            condition = col("dest_location_type") == dest_type
            if dest_condition is None:
                dest_condition = condition
            else:
                dest_condition = dest_condition | condition
        filter_conditions.append(dest_condition)

    else:
        # No specific H/W filter, but might want to exclude certain types
        # Check if we need to exclude origin or dest from being 'H'
        # For nonH: origin != 'H'
        # For all: no additional filter
        pass

    # Apply filters
    df_trips = df_with_next.filter(*filter_conditions)

    # Same day filter
    if trip_filter.same_day_only:
        df_trips = (
            df_trips
            .withColumn("origin_end_date", F.to_date(F.from_unixtime(col("end_timestamp"))))
            .withColumn("dest_start_date", F.to_date(F.from_unixtime(col("dest_start_epoch"))))
            .filter(col("origin_end_date") == col("dest_start_date"))
        )

    # Create sampling UDF
    sample_udf = create_sampling_udf(config.sampling)

    # Sample departure time
    df_trips = (
        df_trips
        .withColumn(
            "rand_start_epoch",
            sample_udf(col("end_timestamp"), col("dest_start_epoch"))
        )
        .withColumn("rand_start_ts", F.to_timestamp(F.from_unixtime(col("rand_start_epoch"))))
        .withColumn("trip_start_hour", hour(col("rand_start_ts")))
    )

    # Select first/last trips if configured
    if trip_filter.select_first_hw or trip_filter.select_last_wh:
        w_first_hw = Window.partitionBy("uid", "origin_end_date").orderBy("rand_start_epoch")
        w_last_wh = Window.partitionBy("uid", "origin_end_date").orderBy(col("rand_start_epoch").desc())

        df_trips = df_trips.withColumn("trip_direction",
            F.when(col("location_type") == "H", "H_to_W").otherwise("W_to_H"))

        if trip_filter.select_first_hw:
            df_trips = df_trips.withColumn("rn_hw",
                F.when(col("trip_direction") == "H_to_W", row_number().over(w_first_hw)).otherwise(999))
        else:
            df_trips = df_trips.withColumn("rn_hw", lit(1))

        if trip_filter.select_last_wh:
            df_trips = df_trips.withColumn("rn_wh",
                F.when(col("trip_direction") == "W_to_H", row_number().over(w_last_wh)).otherwise(999))
        else:
            df_trips = df_trips.withColumn("rn_wh", lit(1))

        df_trips = df_trips.filter((col("rn_hw") == 1) | (col("rn_wh") == 1))
        df_trips = df_trips.drop("rn_hw", "rn_wh", "trip_direction")

    # Keep only necessary columns
    df_trips_slim = (
        df_trips
        .select("uid", "rand_start_ts", "trip_start_hour")
        .withColumn("month", lit(month))
    )

    # Persist and count
    df_trips_slim = df_trips_slim.persist(StorageLevel.DISK_ONLY)
    trips_in_month = df_trips_slim.count()
    print(f"Trips in month {month}: {trips_in_month:,}")

    # Write to parquet
    (df_trips_slim
        .repartition(config.repartition_size)
        .write
        .mode("append")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", config.max_records_per_file)
        .partitionBy("month")
        .parquet(f"{config.temp_dir}/trips"))

    df_trips_slim.unpersist()
    df_month.unpersist()


def compute_hourly_distribution(
    spark: SparkSession,
    config: AnalysisConfig
) -> pd.DataFrame:
    """Compute hourly trip distribution from processed trips.

    Args:
        spark: Spark session
        config: Analysis configuration

    Returns:
        Pandas DataFrame with columns: trip_start_hour, trip_count, percentage_of_trips
    """
    print("\nComputing hourly distribution...")

    # Read all trips
    df_all_trips = spark.read.parquet(f"{config.temp_dir}/trips")
    total_trips = df_all_trips.count()
    print(f"Total trips across all months: {total_trips:,}")

    # Count trips by hour
    trips_by_hour = df_all_trips.groupBy("trip_start_hour") \
        .agg(count("*").alias("trip_count"))

    # Ensure all hours 0-23 are present
    hours_df = spark.createDataFrame([(i,) for i in range(24)], ["trip_start_hour"])
    result_complete = hours_df.join(
        trips_by_hour,
        on="trip_start_hour",
        how="left"
    ).fillna(0).orderBy("trip_start_hour")

    # Convert to pandas
    result_pandas = result_complete.toPandas()
    result_pandas["percentage_of_trips"] = (
        result_pandas["trip_count"] / result_pandas["trip_count"].sum()
    ) * 100

    return result_pandas


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_single_distribution(
    result_df: pd.DataFrame,
    config: AnalysisConfig
) -> None:
    """Plot single departure time distribution.

    Args:
        result_df: DataFrame with columns: trip_start_hour, percentage_of_trips
        config: Analysis configuration
    """
    print("Creating single distribution plot...")

    plt.figure(figsize=(14, 7))
    plt.plot(result_df['trip_start_hour'],
             result_df['percentage_of_trips'],
             marker='o',
             linewidth=2.5,
             markersize=8,
             color='#2E86AB',
             markerfacecolor='#2E86AB',
             markeredgecolor='white',
             markeredgewidth=1.5)

    plt.xlabel('Hour of the Day', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Trips', fontsize=14, fontweight='bold')

    title = f'Percentage of Trips by Hour of the Day - {config.country.upper()}'
    if config.plot_title_suffix:
        title += f' - {config.plot_title_suffix}'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    plt.xticks(range(0, 24, 1), fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim(-0.5, 23.5)
    plt.ylim(0, max(result_df['percentage_of_trips']) * 1.1)

    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    plt.gca().set_axisbelow(True)
    plt.gca().set_facecolor('#F8F9FA')
    plt.gcf().patch.set_facecolor('white')

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.2)

    plt.tight_layout()

    output_path = Path(config.output_dir) / f"trips_by_hour_{config.country}_2023.png"
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two series.

    Args:
        x: First time series
        y: Second time series

    Returns:
        DTW distance
    """
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return D[n, m]


def plot_comparison(
    lbs_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    config: AnalysisConfig
) -> Tuple[float, float]:
    """Plot comparison between LBS data and survey data.

    Args:
        lbs_df: LBS data with columns: trip_start_hour, percentage_of_trips
        survey_df: Survey data with columns: hour, percentage
        config: Analysis configuration

    Returns:
        Tuple of (dtw_distance, dtw_normalized)
    """
    print("Creating comparison plot...")

    # Prepare survey data
    survey_df = survey_df.copy()
    survey_df['hour'] = survey_df['hour'].astype(int)
    survey_df = survey_df.sort_values('hour')

    # Align both series on hours 0..23
    hours = np.arange(24)
    y_lbs = np.interp(hours, lbs_df['trip_start_hour'].values,
                     lbs_df['percentage_of_trips'].values)
    y_survey = np.interp(hours, survey_df['hour'].values,
                        survey_df['percentage'].values)

    # Compute DTW
    dtw_val = dtw_distance(y_lbs, y_survey)
    dtw_norm = dtw_val / (len(y_lbs) + len(y_survey))

    print(f"  DTW = {dtw_val:.3f} | DTW_norm = {dtw_norm:.3f}")

    # Create plot
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    color_lbs = '#2E86AB'
    color_survey = '#D1495B'

    ax.plot(
        lbs_df['trip_start_hour'],
        lbs_df['percentage_of_trips'],
        marker='o', linewidth=3.5, markersize=9,
        color=color_lbs, markerfacecolor=color_lbs,
        markeredgecolor='white', markeredgewidth=1.8,
        label='LBS Data'
    )

    ax.plot(
        survey_df['hour'],
        survey_df['percentage'],
        marker='s', linewidth=3.5, markersize=9,
        color=color_survey, markerfacecolor=color_survey,
        markeredgecolor='white', markeredgewidth=1.8,
        label=f'OD Survey {config.country.upper()}'
    )

    ax.set_xlabel('Hour of Day', fontsize=18, fontweight='bold', labelpad=10)
    ax.set_ylabel('Percentage of Trips', fontsize=18, fontweight='bold', labelpad=10)

    title = f'Percentage of Trips by Hour of the Day - {config.country.upper()}'
    if config.plot_title_suffix:
        title += f' - {config.plot_title_suffix}'
    ax.set_title(title, fontsize=22, fontweight='bold', pad=22)

    ax.set_xlim(-0.5, 23.5)
    ymax = max(
        np.nanmax(lbs_df['percentage_of_trips']),
        np.nanmax(survey_df['percentage'])
    )
    ax.set_ylim(0, ymax * 1.12)

    ax.set_xticks(range(0, 24, 1))
    ax.tick_params(axis='both', labelsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:.0f}%'))

    # Legend with DTW
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color='none'))
    labels.append(f"DTW = {dtw_norm:.3f} (normalized)")
    ax.legend(handles, labels, loc='upper left', frameon=True, fontsize=14,
              title='Series', title_fontsize=15)

    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FAFAFA')
    plt.gcf().patch.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.3)

    plt.tight_layout()

    output_path = Path(config.output_dir) / f"trips_by_hour_comparison_{config.country}_2023.png"
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")

    return dtw_val, dtw_norm


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_analysis(
    config: AnalysisConfig,
    recompute_trips: bool = False
) -> pd.DataFrame:
    """Run complete departure time analysis pipeline.

    Args:
        config: Analysis configuration
        recompute_trips: If True, recompute trip processing even if temp files exist

    Returns:
        DataFrame with hourly distribution results
    """
    print(f"\n{'='*60}")
    print(f"Starting Departure Time Analysis: {config.name}")
    print(f"Country: {config.country}")
    print(f"Sampling strategy: {config.sampling.strategy}")
    print(f"{'='*60}\n")

    # Create Spark session
    spark = create_spark_session(config)

    # Check if we need to process trips
    trips_path = Path(config.temp_dir) / "trips"
    if recompute_trips or not trips_path.exists():
        print(f"Processing trips (saving to {config.temp_dir})...")

        # Load data
        df_clean = spark.read.parquet(config.parquet_dir_in)

        if config.trip_filter.exclude_other:
            df_clean = df_clean.filter(col("location_type") != "O")

        total_rows = df_clean.count()
        print(f"Total rows: {total_rows:,}")

        months = df_clean.select("month").distinct().orderBy("month").collect()
        print(f"Months available: {[m['month'] for m in months]}")

        # Determine which months to process
        if config.month_range:
            months_to_process = months[config.month_range[0]:config.month_range[1]]
        else:
            months_to_process = months

        print(f"Months to process: {[m['month'] for m in months_to_process]}")

        # Process each month
        for month_row in months_to_process:
            month = month_row['month']
            process_trips_for_month(spark, df_clean, month, config)

    else:
        print(f"Using existing processed trips from {config.temp_dir}")

    # Compute hourly distribution
    result_df = compute_hourly_distribution(spark, config)

    # Plot single distribution
    plot_single_distribution(result_df, config)

    # Plot comparison with survey if available
    if config.survey_distribution_path:
        survey_df = pd.read_csv(config.survey_distribution_path)
        dtw_val, dtw_norm = plot_comparison(result_df, survey_df, config)
    else:
        print("\nSkipping survey comparison (no survey path provided)")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Figures saved to: {config.output_dir}")
    print(f"{'='*60}\n")

    # Save results
    output_csv = Path(config.output_dir) / f"hourly_distribution_{config.name}.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    return result_df


if __name__ == "__main__":
    # Example usage
    from config_template import get_config

    config = get_config("HW_informed")
    results = run_analysis(config)
