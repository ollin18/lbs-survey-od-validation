#!/usr/bin/env python
"""
Runner script for departure time analysis.

This script provides a convenient interface for running departure time analyses
with different configurations. It supports:
- Running individual analyses
- Running multiple analyses in sequence
- Recomputing trips or using cached results
- Custom configurations

Usage:
    # Run a single analysis
    python run_analysis.py --config HW_informed

    # Run multiple analyses
    python run_analysis.py --config HW_informed HW_random_uniform HW_raw

    # Run all available analyses
    python run_analysis.py --all

    # Force recomputation of trips (don't use cached results)
    python run_analysis.py --config HW_informed --recompute

    # Use custom country
    python run_analysis.py --config HW_informed --country US

    # List available configurations
    python run_analysis.py --list

Examples:
    # Compare all sampling strategies for Home-Work trips
    python run_analysis.py --config HW_informed HW_random_uniform HW_raw

    # Run all trip type analyses with informed sampling
    python run_analysis.py --config HW_informed HO_informed nonH_informed all_informed

    # Run everything
    python run_analysis.py --all
"""

import argparse
import sys
from pathlib import Path
from typing import List

from config_template import get_config, list_configs, CONFIGS
from departure_time_analysis import run_analysis


def run_single_analysis(config_name: str, country: str, recompute: bool) -> bool:
    """Run a single analysis configuration.

    Args:
        config_name: Name of configuration to run
        country: Country code
        recompute: Whether to recompute trips

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'#'*80}")
        print(f"# Running analysis: {config_name}")
        print(f"{'#'*80}\n")

        config = get_config(config_name, country=country)
        run_analysis(config, recompute_trips=recompute)

        return True

    except Exception as e:
        print(f"\n{'!'*80}")
        print(f"ERROR running {config_name}: {str(e)}")
        print(f"{'!'*80}\n")
        return False


def run_multiple_analyses(
    config_names: List[str],
    country: str,
    recompute: bool
) -> None:
    """Run multiple analyses in sequence.

    Args:
        config_names: List of configuration names
        country: Country code
        recompute: Whether to recompute trips
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"Running {len(config_names)} analyses for {country}")
    print(f"Configurations: {', '.join(config_names)}")
    print(f"{'='*80}\n")

    for config_name in config_names:
        success = run_single_analysis(config_name, country, recompute)
        results[config_name] = "SUCCESS" if success else "FAILED"

    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    for config_name, status in results.items():
        status_symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{status_symbol} {config_name}: {status}")
    print(f"{'='*80}\n")

    # Exit with error code if any failed
    if "FAILED" in results.values():
        sys.exit(1)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run departure time analysis for different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config",
        nargs="+",
        help="Configuration name(s) to run (e.g., HW_informed, HW_random_uniform)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available configurations"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations and exit"
    )

    parser.add_argument(
        "--country",
        default="MX",
        help="Country code (default: MX)"
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of trips (don't use cached results)"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print("\nAvailable configurations:")
        print("=" * 60)
        for config_name in list_configs():
            config = get_config(config_name, country=args.country)
            print(f"\n{config_name}:")
            print(f"  Sampling: {config.sampling.strategy}")
            if config.trip_filter.origin_types and config.trip_filter.dest_types:
                trip_type = f"{config.trip_filter.origin_types[0]}->{config.trip_filter.dest_types[0]}"
            elif config.trip_filter.origin_types:
                trip_type = f"{config.trip_filter.origin_types[0]}->Any"
            else:
                trip_type = "All trips"
            print(f"  Trip type: {trip_type}")
            print(f"  Output: {config.output_dir}")
        print("\n")
        return

    # Determine which configurations to run
    if args.all:
        configs_to_run = list_configs()
    elif args.config:
        configs_to_run = args.config
    else:
        parser.error("Must specify either --config, --all, or --list")

    # Validate configurations
    invalid_configs = [c for c in configs_to_run if c not in CONFIGS]
    if invalid_configs:
        print(f"Error: Unknown configuration(s): {', '.join(invalid_configs)}")
        print(f"Available configurations: {', '.join(list_configs())}")
        print("Use --list to see details")
        sys.exit(1)

    # Run analyses
    if len(configs_to_run) == 1:
        success = run_single_analysis(configs_to_run[0], args.country, args.recompute)
        sys.exit(0 if success else 1)
    else:
        run_multiple_analyses(configs_to_run, args.country, args.recompute)


if __name__ == "__main__":
    main()
