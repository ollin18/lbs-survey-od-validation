"""
Origin-Destination Survey Analysis Tool

A modular and reproducible tool for analyzing OD data from LBS (Location-Based Services)
and comparing it with survey data. This tool can be easily adapted for different cities
by changing the configuration parameters.

Usage:
    python od_survey_analysis.py --city cdmx

Or import and use programmatically:
    from od_survey_analysis import AnalysisConfig, run_analysis
    config = AnalysisConfig(city_name="cdmx", ...)
    run_analysis(config)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import argparse


@dataclass
class AnalysisConfig:
    """Configuration for OD analysis.

    Attributes:
        city_name: Name of the city (used for figure titles and file paths)
        od_data_path: Path to OD pairs CSV file
        geometry_path: Path to geometries GeoJSON file
        survey_od_path: Path to survey OD matrix CSV file (optional - if None, survey analysis will be skipped)
        output_dir: Directory for saving figures
        rounding_factor: Factor for rounding expansion factors (default: 5)
        figure_dpi: DPI for saved figures (default: 300)
    """
    city_name: str
    od_data_path: str
    geometry_path: str
    survey_od_path: Optional[str]
    output_dir: str
    rounding_factor: int = 5
    figure_dpi: int = 300

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def round_to_nearest(x: float, factor: int = 5) -> float:
    """Round a number to the nearest multiple of factor.

    Args:
        x: Number to round
        factor: Rounding factor

    Returns:
        Rounded number
    """
    return np.round(x / factor) * factor


def load_data(config: AnalysisConfig) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Load OD data and geometries.

    Args:
        config: Analysis configuration

    Returns:
        Tuple of (od_dataframe, geometries_geodataframe)
    """
    print(f"Loading data for {config.city_name}...")

    df = pd.read_csv(
        config.od_data_path,
        dtype={"home_geomid": str, "work_geomid": str}
    )
    gdf = gpd.read_file(config.geometry_path)

    print(f"  Loaded {len(df)} OD pairs and {len(gdf)} geometries")
    return df, gdf


def calculate_expansion_factors(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Calculate expansion factors based on population and unique users.

    Args:
        df: OD data with home_geomid and count_uid columns
        gdf: Geometries with geomid and population columns

    Returns:
        GeoDataFrame with expansion factors added
    """
    print("Calculating expansion factors...")

    # Aggregate users by home location
    users = df.groupby("home_geomid")["count_uid"].sum().reset_index()

    # Merge with geometries and calculate expansion factor
    population = gdf.merge(
        users,
        left_on="geomid",
        right_on="home_geomid",
        how="left"
    )
    population["expansion"] = population["population"] / population["count_uid"]
    population["scaled_users"] = population["count_uid"] * population["expansion"]

    print(f"  Calculated expansion factors for {len(population)} districts")
    return population


def plot_expansion_distribution(
    population: gpd.GeoDataFrame,
    config: AnalysisConfig
) -> None:
    """Plot distribution of expansion factors.

    Args:
        population: GeoDataFrame with expansion column
        config: Analysis configuration
    """
    print("Creating expansion factor distribution plot...")

    # Round expansion factors
    population_copy = population.copy()
    population_copy['expansion_rounded'] = population_copy['expansion'].apply(
        lambda x: round_to_nearest(x, config.rounding_factor)
    )

    # Calculate probabilities
    expansion_counts = population_copy['expansion_rounded'].value_counts().sort_index()
    total_count = expansion_counts.sum()
    expansion_probabilities = expansion_counts / total_count

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=expansion_probabilities.index,
        y=expansion_probabilities.values,
        marker='o'
    )
    plt.title(f'Probability Distribution of Expansion Factors - {config.city_name.upper()}')
    plt.xlabel('Expansion Factor')
    plt.ylabel('P(expansion factor)')

    output_path = Path(config.output_dir) / "district_expansion_factor_distribution.png"
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_maps(population: gpd.GeoDataFrame, config: AnalysisConfig) -> None:
    """Create choropleth maps for expansion, population, and user counts.

    Args:
        population: GeoDataFrame with expansion, population, and count_uid columns
        config: Analysis configuration
    """
    maps = [
        ("expansion", "Expansion Factor", "district_expansion_factor_map.png"),
        ("population", "Population", "district_population_map.png"),
        ("count_uid", "Unique Users (LBS)", "district_lbs_uid_map.png"),
    ]

    for column, title, filename in maps:
        print(f"Creating {title} map...")

        fig, ax = plt.subplots(figsize=(10, 10))
        population.plot(column=column, cmap="OrRd", legend=True, ax=ax)
        ax.set_title(f'{title} - {config.city_name.upper()}')
        ax.axis('off')

        output_path = Path(config.output_dir) / filename
        plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved to {output_path}")


def plot_population_scaling(population: gpd.GeoDataFrame, config: AnalysisConfig) -> None:
    """Plot comparison of scaled users vs population.

    Args:
        population: GeoDataFrame with population, count_uid, and scaled_users columns
        config: Analysis configuration
    """
    print("Creating population scaling scatter plot...")

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=population,
        x='count_uid',
        y='population',
        color="blue",
        label='Original Users',
        alpha=0.6
    )
    sns.scatterplot(
        data=population,
        x='scaled_users',
        y='population',
        color="red",
        label='Scaled Users',
        alpha=0.6
    )

    # Add identity line
    min_val = min(population['count_uid'].min(), population['population'].min())
    max_val = max(population['scaled_users'].max(), population['population'].max())
    identity_line = np.linspace(min_val, max_val, 100)
    plt.plot(identity_line, identity_line, 'k--', label='Identity Line', alpha=0.5)

    plt.title(f'Population vs. Unique Users - {config.city_name.upper()}')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Total Population')
    plt.xlabel('Unique Users')
    plt.legend()

    output_path = Path(config.output_dir) / "district_scaled_population_scatter.png"
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def create_od_matrix(
    df: pd.DataFrame,
    population: gpd.GeoDataFrame,
    exclude_zero: bool = True
) -> pd.DataFrame:
    """Create OD matrix from location-based services data.

    Args:
        df: OD pairs with home_geomid, work_geomid, and count_uid columns
        population: GeoDataFrame with expansion factors
        exclude_zero: Whether to exclude geomid '0' (default: True)

    Returns:
        Wide-format OD matrix
    """
    print("Creating LBS OD matrix...")

    # Prepare home-work data with expansion factors
    hw_u = df.dropna(subset=["home_geomid", "work_geomid"])
    hw_u = pd.merge(
        hw_u,
        population[["geomid", "expansion"]],
        left_on="home_geomid",
        right_on="geomid",
        how="left"
    ).rename(columns={"expansion": "home_expansion"}).drop(columns=["geomid"])

    hw_u = hw_u[["count_uid", "home_geomid", "work_geomid", "home_expansion"]]

    # Get all unique geomids
    unique_geomid = list(
        set(hw_u["home_geomid"].values) | set(hw_u["work_geomid"].values)
    )
    full_geomid = sorted(unique_geomid)

    # Create expanded counts
    geomid_counts = hw_u.copy()
    if exclude_zero:
        geomid_counts = geomid_counts.query("home_geomid != '0'").query("work_geomid != '0'")

    geomid_counts['expanded_count'] = (
        geomid_counts['count_uid'] * geomid_counts['home_expansion']
    )
    geomid_counts = geomid_counts.groupby(
        ['home_geomid', 'work_geomid']
    )["expanded_count"].sum().reset_index(name='counts')

    # Pivot to wide format
    wide_od = geomid_counts.pivot(
        index='home_geomid',
        columns='work_geomid',
        values='counts'
    )
    wide_od = wide_od.reindex(
        index=full_geomid,
        columns=full_geomid,
        fill_value=0
    ).fillna(0)

    # Remove '0' if it exists
    if exclude_zero:
        if '0' in wide_od.index:
            wide_od = wide_od.drop(index=['0'])
        if '0' in wide_od.columns:
            wide_od = wide_od.drop(columns=['0'])

    print(f"  Created {wide_od.shape[0]}x{wide_od.shape[1]} OD matrix")
    return wide_od, geomid_counts


def load_survey_od_matrix(
    config: AnalysisConfig,
    reference_geomids: list,
    exclude_zero: bool = True
) -> pd.DataFrame:
    """Load and format survey OD matrix.

    Args:
        config: Analysis configuration
        reference_geomids: List of geomids to align with
        exclude_zero: Whether to exclude geomid '0' (default: True)

    Returns:
        Wide-format survey OD matrix
    """
    print("Loading survey OD matrix...")

    survey_od = pd.read_csv(
        config.survey_od_path,
        dtype={"home_geomid": str, "work_geomid": str}
    )

    wide_survey_od = survey_od.pivot(
        index='home_geomid',
        columns='work_geomid',
        values='counts'
    )
    wide_survey_od = wide_survey_od.reindex(
        index=reference_geomids,
        columns=reference_geomids,
        fill_value=0
    ).fillna(0)

    # Remove '0' if it exists
    if exclude_zero:
        if '0' in wide_survey_od.index:
            wide_survey_od = wide_survey_od.drop(index=['0'])
        if '0' in wide_survey_od.columns:
            wide_survey_od = wide_survey_od.drop(columns=['0'])

    print(f"  Loaded {wide_survey_od.shape[0]}x{wide_survey_od.shape[1]} survey OD matrix")
    return wide_survey_od, survey_od


def plot_od_heatmap(
    od_matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    config: AnalysisConfig
) -> None:
    """Plot OD matrix as heatmap.

    Args:
        od_matrix: Wide-format OD matrix
        title: Plot title
        output_path: Path to save figure
        config: Analysis configuration
    """
    print(f"Creating {title} heatmap...")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        od_matrix,
        cmap='magma',
        linewidths=0,
        norm=plt.matplotlib.colors.LogNorm(),
        cbar_kws={'label': 'Counts (log scale)'}
    )

    plt.title(f'{title} - {config.city_name.upper()}')
    plt.xlabel('Work District')
    plt.ylabel('Home District')
    plt.tight_layout()

    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def compare_od_matrices(
    lbs_od_long: pd.DataFrame,
    survey_od_long: pd.DataFrame,
    config: AnalysisConfig
) -> None:
    """Compare LBS and survey OD matrices with correlation analysis.

    Args:
        lbs_od_long: Long-format LBS OD data
        survey_od_long: Long-format survey OD data
        config: Analysis configuration
    """
    print("Comparing LBS and survey OD matrices...")

    # Merge datasets
    both_long = pd.merge(
        lbs_od_long,
        survey_od_long,
        on=["home_geomid", "work_geomid"],
        suffixes=("_lbs", "_eod")
    )

    # Separate intra and inter-district flows
    intra = both_long[both_long["home_geomid"] == both_long["work_geomid"]]
    inter = both_long[both_long["home_geomid"] != both_long["work_geomid"]]

    # Calculate correlations on log scale
    overall_corr = np.log10(both_long['counts_lbs']).corr(
        np.log10(both_long['counts_eod'])
    )
    intra_corr = np.log10(intra['counts_lbs']).corr(
        np.log10(intra['counts_eod'])
    )
    inter_corr = np.log10(inter['counts_lbs']).corr(
        np.log10(inter['counts_eod'])
    )

    print(f"  Overall correlation: {overall_corr:.3f}")
    print(f"  Intra-district correlation: {intra_corr:.3f}")
    print(f"  Inter-district correlation: {inter_corr:.3f}")

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        data=intra,
        x='counts_lbs',
        y='counts_eod',
        color="blue",
        label='Intra-District',
        alpha=0.6
    )
    sns.scatterplot(
        data=inter,
        x='counts_lbs',
        y='counts_eod',
        color="red",
        label='Inter-District',
        alpha=0.6
    )

    # Add identity line
    min_val = np.min([both_long['counts_lbs'].min(), both_long['counts_eod'].min()]) - 1
    max_val = np.max([both_long['counts_lbs'].max(), both_long['counts_eod'].max()])
    identity_line = np.linspace(min_val, max_val, 100)
    plt.plot(identity_line, identity_line, 'k--', label='Identity', alpha=0.5)

    # Set log scale and limits
    plt.xscale('log')
    plt.yscale('log')

    log_min_val = np.log10(min_val)
    log_max_val = np.log10(max_val)
    plt.xlim(10**log_min_val, 10**log_max_val)
    plt.ylim(10**log_min_val, 10**log_max_val)

    plt.ylabel('Survey OD Counts')
    plt.xlabel('LBS OD Counts')
    plt.title(f'OD Comparison: LBS vs Survey - {config.city_name.upper()}')

    # Add correlation text box
    plt.text(
        0.05, 0.95,
        f'Overall: r = {overall_corr:.3f}\nIntra: r = {intra_corr:.3f}\nInter: r = {inter_corr:.3f}',
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.legend()

    output_path = Path(config.output_dir) / "district_lbs_survey_correlation_scatter.png"
    plt.savefig(output_path, dpi=config.figure_dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def run_analysis(config: AnalysisConfig) -> None:
    """Run complete OD analysis pipeline.

    Args:
        config: Analysis configuration
    """
    print(f"\n{'='*60}")
    print(f"Starting OD Analysis for {config.city_name.upper()}")
    print(f"{'='*60}\n")

    # Load data
    df, gdf = load_data(config)

    # Calculate expansion factors
    population = calculate_expansion_factors(df, gdf)

    # Create expansion factor visualizations
    plot_expansion_distribution(population, config)
    plot_maps(population, config)
    plot_population_scaling(population, config)

    # Create LBS OD matrix
    wide_lbs_od, lbs_od_long = create_od_matrix(df, population)
    plot_od_heatmap(
        wide_lbs_od,
        "LBS OD Matrix",
        Path(config.output_dir) / "district_lbs_OD_heatmap.png",
        config
    )

    # Load and plot survey OD matrix (only if survey path is provided)
    if config.survey_od_path is not None:
        wide_survey_od, survey_od_long = load_survey_od_matrix(
            config,
            wide_lbs_od.index.tolist()
        )
        plot_od_heatmap(
            wide_survey_od,
            "Survey OD Matrix",
            Path(config.output_dir) / "district_survey_OD_heatmap.png",
            config
        )

        # Compare OD matrices
        compare_od_matrices(lbs_od_long, survey_od_long, config)
    else:
        print("\nSkipping survey analysis (no survey path provided)")

    print(f"\n{'='*60}")
    print(f"Analysis complete! Figures saved to: {config.output_dir}")
    print(f"{'='*60}\n")


def create_config_for_cdmx() -> AnalysisConfig:
    """Create configuration for CDMX analysis.

    Returns:
        AnalysisConfig for CDMX
    """
    return AnalysisConfig(
        city_name="cdmx",
        od_data_path="../../data/intermediate/od_pairs/cdmx_od_geomid.csv",
        geometry_path="../../data/intermediate/geometries/cdmx_geometries.geojson",
        survey_od_path="../../data/clean/cdmx/survey/od_matrix.csv",
        output_dir="../../figures/cdmx",
        rounding_factor=5,
        figure_dpi=300
    )


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run OD analysis for a city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis for CDMX with default paths
  python od_survey_analysis.py --city cdmx

  # Run analysis with custom paths
  python od_survey_analysis.py --city cdmx \\
      --od-data path/to/od.csv \\
      --geometry path/to/geo.geojson \\
      --survey-od path/to/survey.csv \\
      --output-dir path/to/figures
        """
    )

    parser.add_argument(
        "--city",
        required=True,
        help="City name (used for titles and default paths)"
    )
    parser.add_argument(
        "--od-data",
        help="Path to OD pairs CSV file"
    )
    parser.add_argument(
        "--geometry",
        help="Path to geometries GeoJSON file"
    )
    parser.add_argument(
        "--survey-od",
        help="Path to survey OD matrix CSV file"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for saving figures"
    )
    parser.add_argument(
        "--rounding-factor",
        type=int,
        default=5,
        help="Rounding factor for expansion factors (default: 5)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)"
    )

    args = parser.parse_args()

    # Use CDMX defaults if specific paths not provided
    if args.city.lower() == "cdmx" and not all([args.od_data, args.geometry, args.survey_od]):
        print("Using default CDMX configuration...")
        config = create_config_for_cdmx()

        if args.od_data:
            config.od_data_path = args.od_data
        if args.geometry:
            config.geometry_path = args.geometry
        if args.survey_od:
            config.survey_od_path = args.survey_od
        if args.output_dir:
            config.output_dir = args.output_dir
    else:
        if not all([args.od_data, args.geometry, args.output_dir]):
            parser.error(
                "For cities other than CDMX, you must provide: "
                "--od-data, --geometry, and --output-dir "
                "(--survey-od is optional)"
            )

        config = AnalysisConfig(
            city_name=args.city,
            od_data_path=args.od_data,
            geometry_path=args.geometry,
            survey_od_path=args.survey_od,
            output_dir=args.output_dir,
            rounding_factor=args.rounding_factor,
            figure_dpi=args.dpi
        )

    run_analysis(config)


if __name__ == "__main__":
    main()
