#!/usr/bin/env python
"""
Compare multiple departure time analyses side-by-side.

This script generates combined plots showing multiple analyses together,
useful for comparing:
- Different sampling strategies (informed, random, raw)
- Different trip types (HW, HO, nonH, all)
- Different configurations

Usage:
    # Compare sampling strategies
    python compare_analyses.py --configs HW_informed HW_random_uniform HW_raw \
        --title "Comparison of Sampling Strategies" \
        --output comparison_sampling.png

    # Compare trip types
    python compare_analyses.py --configs HW_informed HO_informed nonH_informed all_informed \
        --title "Comparison of Trip Types" \
        --output comparison_trip_types.png

    # Load from saved results (faster)
    python compare_analyses.py --results-dir ../../../figures/MX/departure_time \
        --configs HW_informed HW_random_uniform HW_raw
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional
import seaborn as sns

from config_template import get_config


def load_results(config_name: str, country: str, results_dir: Optional[str] = None) -> pd.DataFrame:
    """Load results for a configuration.

    Args:
        config_name: Name of configuration
        country: Country code
        results_dir: Optional directory containing saved results. If None, uses config output_dir

    Returns:
        DataFrame with hourly distribution
    """
    if results_dir:
        csv_path = Path(results_dir) / config_name / f"hourly_distribution_{config_name}.csv"
    else:
        config = get_config(config_name, country=country)
        csv_path = Path(config.output_dir) / f"hourly_distribution_{config_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results not found for {config_name}. "
            f"Run the analysis first: python run_analysis.py --config {config_name}"
        )

    return pd.read_csv(csv_path)


def plot_comparison(
    results: dict,
    title: str,
    output_path: str,
    country: str,
    include_survey: bool = True,
    survey_path: Optional[str] = None,
    figsize: tuple = (16, 8)
) -> None:
    """Create comparison plot for multiple analyses.

    Args:
        results: Dictionary mapping config names to DataFrames
        title: Plot title
        output_path: Path to save figure
        country: Country code
        include_survey: Whether to include survey data
        survey_path: Path to survey distribution CSV
        figsize: Figure size
    """
    # Set up color palette
    n_configs = len(results)
    if include_survey and survey_path:
        n_configs += 1

    colors = sns.color_palette("husl", n_configs)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot each configuration
    for i, (config_name, df) in enumerate(results.items()):
        ax.plot(
            df['trip_start_hour'],
            df['percentage_of_trips'],
            marker='o',
            linewidth=3,
            markersize=8,
            color=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor='white',
            markeredgewidth=1.5,
            label=config_name,
            alpha=0.8
        )

    # Plot survey data if available
    if include_survey and survey_path and Path(survey_path).exists():
        survey_df = pd.read_csv(survey_path)
        ax.plot(
            survey_df['hour'],
            survey_df['percentage'],
            marker='s',
            linewidth=3,
            markersize=8,
            color='black',
            markerfacecolor='black',
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='Survey Data',
            alpha=0.8,
            linestyle='--'
        )

    # Formatting
    ax.set_xlabel('Hour of Day', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Percentage of Trips (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_title(f'{title} - {country.upper()}', fontsize=20, fontweight='bold', pad=20)

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 1))
    ax.tick_params(axis='both', labelsize=13)

    # Legend
    ax.legend(loc='best', frameon=True, fontsize=12, title='Configuration', title_fontsize=13)

    # Grid and aesthetics
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FAFAFA')
    plt.gcf().patch.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def plot_heatmap_comparison(
    results: dict,
    title: str,
    output_path: str,
    country: str
) -> None:
    """Create heatmap showing all distributions.

    Args:
        results: Dictionary mapping config names to DataFrames
        title: Plot title
        output_path: Path to save figure
        country: Country code
    """
    # Prepare data matrix
    hours = range(24)
    data = []
    labels = []

    for config_name, df in results.items():
        data.append(df['percentage_of_trips'].values)
        labels.append(config_name)

    data_matrix = np.array(data)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, len(labels) * 0.8 + 2))

    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Labels
    ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title(f'{title} - {country.upper()}', fontsize=16, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage of Trips (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(labels)):
        for j in range(24):
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def compute_statistics(results: dict) -> pd.DataFrame:
    """Compute comparison statistics.

    Args:
        results: Dictionary mapping config names to DataFrames

    Returns:
        DataFrame with statistics
    """
    stats = []

    for config_name, df in results.items():
        pct = df['percentage_of_trips'].values
        hours = df['trip_start_hour'].values

        # Weighted mean hour
        mean_hour = np.average(hours, weights=pct)

        # Peak hour
        peak_hour = hours[np.argmax(pct)]
        peak_pct = np.max(pct)

        # Standard deviation
        std_hour = np.sqrt(np.average((hours - mean_hour)**2, weights=pct))

        stats.append({
            'Configuration': config_name,
            'Mean Hour': f'{mean_hour:.2f}',
            'Std Dev': f'{std_hour:.2f}',
            'Peak Hour': int(peak_hour),
            'Peak %': f'{peak_pct:.2f}%'
        })

    return pd.DataFrame(stats)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare multiple departure time analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Configuration names to compare"
    )

    parser.add_argument(
        "--title",
        default="Departure Time Comparison",
        help="Plot title"
    )

    parser.add_argument(
        "--output",
        default="comparison.png",
        help="Output filename (default: comparison.png)"
    )

    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Also generate heatmap plot"
    )

    parser.add_argument(
        "--country",
        default="MX",
        help="Country code (default: MX)"
    )

    parser.add_argument(
        "--results-dir",
        help="Directory containing saved results (optional)"
    )

    parser.add_argument(
        "--no-survey",
        action="store_true",
        help="Don't include survey data in comparison"
    )

    parser.add_argument(
        "--survey-path",
        default="/global/home/users/ollin/test_cuebiq/informed_versions/hw_trips_by_hour.csv",
        help="Path to survey distribution CSV"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print comparison statistics"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results for: {', '.join(args.configs)}")
    results = {}
    for config_name in args.configs:
        try:
            df = load_results(config_name, args.country, args.results_dir)
            results[config_name] = df
            print(f"  ✓ Loaded {config_name}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            continue

    if not results:
        print("\nNo results loaded. Run analyses first:")
        print(f"  python run_analysis.py --config {' '.join(args.configs)}")
        return

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(
        results,
        args.title,
        args.output,
        args.country,
        include_survey=not args.no_survey,
        survey_path=args.survey_path if not args.no_survey else None
    )

    # Generate heatmap if requested
    if args.heatmap:
        print("Generating heatmap...")
        heatmap_path = args.output.replace('.png', '_heatmap.png')
        plot_heatmap_comparison(results, args.title, heatmap_path, args.country)

    # Print statistics if requested
    if args.stats:
        print("\nComparison Statistics:")
        print("=" * 80)
        stats_df = compute_statistics(results)
        print(stats_df.to_string(index=False))
        print("=" * 80)

        # Save statistics
        stats_path = args.output.replace('.png', '_stats.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics saved to: {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
