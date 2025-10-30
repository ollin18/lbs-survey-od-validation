"""
Configuration Template for OD Analysis

Copy this file and modify the parameters for your city.
Then run: python your_config.py
"""

from od_survey_analysis import AnalysisConfig, run_analysis

# =============================================================================
# MODIFY THESE PARAMETERS FOR YOUR CITY
# =============================================================================

config = AnalysisConfig(
    # City name (used in figure titles)
    city_name="your_city_name",

    # Path to OD pairs CSV file
    # Required columns: home_geomid, work_geomid, count_uid
    od_data_path="../../data/intermediate/od_pairs/your_city_od_geomid.csv",

    # Path to geometries GeoJSON file
    # Required fields: geomid, population, geometry
    geometry_path="../../data/intermediate/geometries/your_city_geometries.geojson",

    # Path to survey OD matrix CSV file (OPTIONAL - set to None to skip survey analysis)
    # Required columns: home_geomid, work_geomid, counts
    survey_od_path="../../data/clean/your_city/survey/od_matrix.csv",
    # Or set to None if no survey data is available:
    # survey_od_path=None,

    # Output directory for figures
    output_dir="../../figures/your_city",

    # Optional parameters (can be omitted to use defaults)
    rounding_factor=5,    # Factor for rounding expansion factors
    figure_dpi=300        # DPI for saved figures
)

# =============================================================================
# RUN ANALYSIS
# =============================================================================

#  if __name__ == "__main__":
#      run_analysis(config)


# =============================================================================
# EXAMPLES FOR SPECIFIC CITIES
# =============================================================================

def cdmx_config():
    """Configuration for CDMX (Mexico City)"""
    return AnalysisConfig(
        city_name="cdmx",
        od_data_path="../../data/intermediate/od_pairs/cdmx_od_geomid.csv",
        geometry_path="../../data/intermediate/geometries/cdmx_geometries.geojson",
        survey_od_path="../../data/clean/cdmx/survey/od_matrix.csv",
        output_dir="../../figures/cdmx",
        rounding_factor=5,
        figure_dpi=300
    )

def example_no_survey():
    """Example configuration without survey data (survey analysis will be skipped)"""
    return AnalysisConfig(
        city_name="example_city",
        od_data_path="../../data/intermediate/od_pairs/example_city_od_geomid.csv",
        geometry_path="../../data/intermediate/geometries/example_city_geometries.geojson",
        survey_od_path=None,  # No survey data available - will skip survey analysis
        output_dir="../../figures/example_city",
        rounding_factor=5,
        figure_dpi=300
    )

def cdmx_ageb():
    """Configuration for CDMX AGEB level analysis"""
    return AnalysisConfig(
        city_name="cdmx_ageb",
        od_data_path="../../data/intermediate/od_pairs/cdmx_agebs.csv",
        geometry_path="../../data/intermediate/geometries/cdmx_agebs_zm.geojson",
        survey_od_path=None,
        output_dir="../../figures/cdmx_ageb",
        rounding_factor=5,
        figure_dpi=300
    )

def guadalajara_ageb():
    """Configuration for Guadalajara AGEB level analysis"""
    return AnalysisConfig(
        city_name="Guadalajara",
        od_data_path="../../data/intermediate/od_pairs/guadalajara_agebs.csv",
        geometry_path="../../data/intermediate/geometries/guadalajara_agebs_zm.geojson",
        survey_od_path=None,
        output_dir="../../figures/guadalajara",
        rounding_factor=5,
        figure_dpi=300
    )


# Uncomment and run this to use CDMX configuration:
# if __name__ == "__main__":
#     run_analysis(cdmx_config())
if __name__ == "__main__":
    run_analysis(cdmx_config())
