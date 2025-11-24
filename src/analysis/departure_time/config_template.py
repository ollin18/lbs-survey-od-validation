"""
Configuration Template for Departure Time Analysis

This file contains all configuration parameters for different departure time analyses.
Each configuration defines:
- Data paths
- Trip type filters (HW, HO, nonH, all)
- Sampling strategy (informed, random-uniform, raw)
- Spark settings
- Output paths

Usage:
    from config_template import get_config
    config = get_config("HW_informed")
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path


@dataclass
class SparkConfig:
    """Spark configuration parameters."""
    app_name: str = "MobilityAnalysis"
    master: str = "local[20]"
    driver_memory: str = "60g"
    driver_max_result_size: str = "20g"
    executor_memory: str = "60g"
    local_dir: str = "/global/scratch/p2p3/pl1_lbs/ollin"
    shuffle_partitions: int = 40
    default_parallelism: int = 40
    memory_fraction: float = 0.8
    storage_fraction: float = 0.3


@dataclass
class TripFilterConfig:
    """Configuration for filtering trips by type."""
    origin_types: Optional[List[str]] = None  # e.g., ['H'], ['W'], None for all
    dest_types: Optional[List[str]] = None    # e.g., ['W'], ['H'], None for all

    weekday_only: bool = True

    exclude_other: bool = True

    same_day_only: bool = True

    select_first_hw: bool = False  # First H->W trip of the day
    select_last_wh: bool = False   # Last W->H trip of the day


@dataclass
class SamplingConfig:
    """Configuration for departure time sampling."""
    # Sampling strategy: 'informed' (uses distribution), 'random_uniform', 'raw' (no sampling)
    strategy: str = "informed"

    # Path to hourly distribution CSV (for 'informed' strategy)
    # CSV should have columns: hour, percentage
    distribution_path: Optional[str] = None

    # Distribution dict (alternative to file, for 'informed' strategy)
    # This allows to directly specify the distribution in code if we don't have
    # a file
    distribution_dict: Optional[Dict[int, float]] = None


@dataclass
class AnalysisConfig:
    """Complete configuration for a departure time analysis."""
    name: str

    country: str
    parquet_dir_in: str
    temp_dir: str
    output_dir: str

    # Survey comparison data (optional)
    survey_distribution_path: Optional[str] = None

    # Month range to process
    month_range: Optional[Tuple[int, int]] = None  # e.g., (1, 3) for months[1:3], None for all

    spark_config: SparkConfig = None
    trip_filter: TripFilterConfig = None
    sampling: SamplingConfig = None

    repartition_size: int = 200
    max_records_per_file: int = 5_000_000

    figure_dpi: int = 300
    plot_title_suffix: str = ""

    def __post_init__(self):
        """Ensure output directory exists and set defaults."""
        if self.spark_config is None:
            self.spark_config = SparkConfig()
        if self.trip_filter is None:
            self.trip_filter = TripFilterConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PREDEFINED CONFIGURATIONS
# =============================================================================

def hw_informed_config(country: str = "MX") -> AnalysisConfig:
    """Home-to-Work trips with informed sampling (Informed distribution)."""
    return AnalysisConfig(
        name="HW_informed",
        country=country,
        parquet_dir_in=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023",
        temp_dir=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_informed/{country}_2023",
        output_dir=f"../../../figures/{country}/departure_time/HW_informed",
        survey_distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        month_range=(1, 3),  # Process months[1:3]
        trip_filter=TripFilterConfig(
            origin_types=['H'],
            dest_types=['W'],
            weekday_only=True,
            exclude_other=True,
            same_day_only=True,
            select_first_hw=True,
            select_last_wh=True
        ),
        sampling=SamplingConfig(
            strategy="informed",
            distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv"
        ),
        plot_title_suffix="Informed"
    )


def hw_random_uniform_config(country: str = "MX") -> AnalysisConfig:
    """Home-to-Work trips with random uniform sampling."""
    return AnalysisConfig(
        name="HW_random_uniform",
        country=country,
        parquet_dir_in=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023",
        temp_dir=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_random_uniform/{country}_2023",
        output_dir=f"../../../figures/{country}/departure_time/HW_random_uniform",
        survey_distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        month_range=(1, 3),
        trip_filter=TripFilterConfig(
            origin_types=['H'],
            dest_types=['W'],
            weekday_only=True,
            exclude_other=True,
            same_day_only=True,
            select_first_hw=True,
            select_last_wh=True
        ),
        sampling=SamplingConfig(
            strategy="random_uniform"
        ),
        plot_title_suffix="Random Uniform"
    )


def hw_raw_config(country: str = "MX") -> AnalysisConfig:
    """Home-to-Work trips with raw data (no sampling)."""
    return AnalysisConfig(
        name="HW_raw",
        country=country,
        parquet_dir_in=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023",
        temp_dir=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_raw/{country}_2023",
        output_dir=f"../../../figures/{country}/departure_time/HW_raw",
        survey_distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        month_range=(1, 3),
        trip_filter=TripFilterConfig(
            origin_types=['H'],
            dest_types=['W'],
            weekday_only=True,
            exclude_other=True,
            same_day_only=True,
            select_first_hw=True,
            select_last_wh=True
        ),
        sampling=SamplingConfig(
            strategy="raw"
        ),
        plot_title_suffix="Raw Data"
    )


def ho_informed_config(country: str = "MX") -> AnalysisConfig:
    """Home-to-Other trips with informed sampling."""
    return AnalysisConfig(
        name="HO_informed",
        country=country,
        parquet_dir_in=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023",
        temp_dir=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_informed_HO/{country}_2023",
        output_dir=f"../../../figures/{country}/departure_time/HO_informed",
        survey_distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        month_range=(1, 3),
        trip_filter=TripFilterConfig(
            origin_types=['H'],
            dest_types=None,  # Any destination except Home
            weekday_only=True,
            exclude_other=True,
            same_day_only=True,
            select_first_hw=False,
            select_last_wh=False
        ),
        sampling=SamplingConfig(
            strategy="informed",
            distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv"
        ),
        plot_title_suffix="Home-to-Other - Informed"
    )


def nonh_informed_config(country: str = "MX") -> AnalysisConfig:
    """Non-Home trips with informed sampling."""
    return AnalysisConfig(
        name="nonH_informed",
        country=country,
        parquet_dir_in=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023",
        temp_dir=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_informed_nonH/{country}_2023",
        output_dir=f"../../../figures/{country}/departure_time/nonH_informed",
        survey_distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        month_range=(1, 3),
        trip_filter=TripFilterConfig(
            origin_types=None,  # Any origin except Home
            dest_types=None,
            weekday_only=True,
            exclude_other=True,
            same_day_only=True,
            select_first_hw=False,
            select_last_wh=False
        ),
        sampling=SamplingConfig(
            strategy="informed",
            distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv"
        ),
        plot_title_suffix="Non-Home - Informed"
    )


def all_informed_config(country: str = "MX") -> AnalysisConfig:
    """All trips with informed sampling."""
    return AnalysisConfig(
        name="all_informed",
        country=country,
        parquet_dir_in=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023",
        temp_dir=f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_informed_all/{country}_2023",
        output_dir=f"../../../figures/{country}/departure_time/all_informed",
        survey_distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        month_range=(1, 3),
        trip_filter=TripFilterConfig(
            origin_types=None,
            dest_types=None,
            weekday_only=True,
            exclude_other=True,
            same_day_only=True,
            select_first_hw=False,
            select_last_wh=False
        ),
        sampling=SamplingConfig(
            strategy="informed",
            distribution_path="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv"
        ),
        plot_title_suffix="All Trips - Informed"
    )


# =============================================================================
# CONFIGURATION REGISTRY
# =============================================================================

CONFIGS = {
    "HW_informed": hw_informed_config,
    "HW_random_uniform": hw_random_uniform_config,
    "HW_raw": hw_raw_config,
    "HO_informed": ho_informed_config,
    "nonH_informed": nonh_informed_config,
    "all_informed": all_informed_config,
}


def get_config(name: str, country: str = "MX") -> AnalysisConfig:
    """Get a predefined configuration by name.

    Args:
        name: Configuration name (e.g., 'HW_informed', 'HW_random_uniform', 'HW_raw')
        country: Country code (default: 'MX')

    Returns:
        AnalysisConfig object

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown configuration: {name}. Available: {available}")

    return CONFIGS[name](country=country)


def list_configs() -> List[str]:
    """List all available configuration names."""
    return list(CONFIGS.keys())


if __name__ == "__main__":
    # Example usage
    print("Available configurations:")
    for config_name in list_configs():
        print(f"  - {config_name}")

    print("\nExample: HW_informed configuration:")
    config = get_config("HW_informed")
    print(f"  Name: {config.name}")
    print(f"  Country: {config.country}")
    print(f"  Temp dir: {config.temp_dir}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Trip filter: H->W, weekday only")
    print(f"  Sampling: {config.sampling.strategy}")
