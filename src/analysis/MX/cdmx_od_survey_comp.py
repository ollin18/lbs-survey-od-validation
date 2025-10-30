#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def round_to_nearest_10(x):
    return np.round(x / 5) * 5

#%%
df = pd.read_csv("../../../data/intermediate/od_pairs/cdmx_od_geomid.csv", dtype={"home_geomid": str, "work_geomid": str})
gdf = gpd.read_file("../../../data/intermediate/geometries/cdmx_geometries.geojson")# %%
users = df.groupby("home_geomid")["count_uid"].sum().reset_index()

# %%
population = gdf.merge(users, left_on="geomid", right_on="home_geomid", how="left")
population["expansion"] = population["population"] / population["count_uid"]


#%%
population['expansionf_r'] = population['expansion'].apply(round_to_nearest_10)
expansionf_counts = population['expansionf_r'].value_counts().sort_index()

total_count = expansionf_counts.sum()
expansionf_probabilities = expansionf_counts / total_count

plt.figure(figsize=(10, 6))
sns.lineplot(x=expansionf_probabilities.index, y=expansionf_probabilities.values, marker='o')
plt.title('Probability Distribution of Expansionf')
plt.xlabel('Expansion Factor')
plt.ylabel('P(ef)')
plt.show()
# Save figure
plt.savefig("../../../figures/cdmx/district_expansion_factor_distribution.png", dpi=300)

# %%
population.plot(column="expansion", cmap="OrRd", legend=True)
plt.savefig("../../../figures/cdmx/district_expansion_factor_map.png", dpi=300)

# %%
population.plot(column="population", cmap="OrRd", legend=True)
plt.savefig("../../../figures/cdmx/district_population_map.png", dpi=300)

# %%
population.plot(column="count_uid", cmap="OrRd", legend=True)
plt.savefig("../../../figures/cdmx/district_lbs_uid_map.png", dpi=300)


# %%
population['scaled_users'] = population['count_uid'] * population['expansion']

# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=population, x='count_uid', y='population', color="blue")
sns.scatterplot(data=population, x='scaled_users', y='population', color="red")
# Add identity line
min_val = min(population['count_uid'].min(), population['population'].min())
max_val = max(population['count_uid'].max(), population['population'].max())
identity_line = np.linspace(min_val, max_val, 100)
plt.plot(identity_line, identity_line, 'r--', label='Identity Line')

plt.title('Scaled Population vs. Unique Users')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Total Population')
plt.xlabel('Unique Users')
plt.legend()
plt.savefig("../../../figures/cdmx/district_scaled_population_scatter.png", dpi=300)
plt.show()


# %%
hw_u = df.dropna(subset=["home_geomid", "work_geomid"])
hw_u = pd.merge(hw_u, population[["geomid", "expansion"]], left_on="home_geomid", right_on="geomid", how="left").rename(columns={"expansion": "home_expansion"}).drop(columns=["geomid"])
hw_u = pd.merge(hw_u, population[["geomid", "expansion"]], left_on="work_geomid", right_on="geomid", how="left").rename(columns={"expansion": "work_expansion"}).drop(columns=["geomid"])
hw_u = hw_u[["count_uid", "home_geomid", "work_geomid", "home_expansion"]]

# %%
unique_geomid = list(set(hw_u["home_geomid"].values) | set(hw_u["work_geomid"].values))
full_geomid = sorted(unique_geomid)  # Create a sorted list of all unique districts

# %%
geomid_counts = hw_u.query("home_geomid != '0'").query("work_geomid != '0'")
geomid_counts['expanded_count'] = geomid_counts['count_uid'] * geomid_counts['home_expansion']
geomid_counts = geomid_counts.groupby(['home_geomid', 'work_geomid'])["expanded_count"].sum().reset_index(name='counts')
wide_lbs_od = geomid_counts.pivot(index='home_geomid', columns='work_geomid', values='counts')
wide_lbs_od = wide_lbs_od.reindex(index=full_geomid, columns=full_geomid, fill_value=0).fillna(0)
# Remove column '0' and row '0' if exists
if '0' in wide_lbs_od.index:
    wide_lbs_od = wide_lbs_od.drop(index=['0'])
if '0' in wide_lbs_od.columns:
    wide_lbs_od = wide_lbs_od.drop(columns=['0'])
wide_lbs_od

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(wide_lbs_od, cmap='magma', linewidths=0, norm=plt.matplotlib.colors.LogNorm())

plt.title('LBS Heatmap of Home and Work Geomid - CDMX')
plt.xlabel('Work Geomid')
plt.ylabel('Home Geomid')
plt.savefig("../../../figures/cdmx/district_lbs_OD_heatmap.png", dpi=300)
plt.show()

# With the survey
# %%
survey_od = pd.read_csv("../../../data/clean/cdmx/survey/od_matrix.csv", dtype={"home_geomid": str, "work_geomid": str})
wide_survey_od = survey_od.pivot(index='home_geomid', columns='work_geomid', values='counts')
wide_survey_od = wide_survey_od.reindex(index=full_geomid, columns=full_geomid, fill_value=0).fillna(0)
if '0' in wide_survey_od.index:
    wide_survey_od = wide_survey_od.drop(index=['0'])
if '0' in wide_survey_od.columns:
    wide_survey_od = wide_survey_od.drop(columns=['0'])
wide_survey_od

# %%
plt.figure(figsize=(12, 10))
sns.heatmap(wide_survey_od, cmap='magma', linewidths=0, norm=plt.matplotlib.colors.LogNorm())

plt.title('Survey Heatmap of Home and Work Geomid - CDMX')
plt.xlabel('Work Geomid')
plt.ylabel('Home Geomid')
plt.savefig("../../../figures/cdmx/district_survey_OD_heatmap.png", dpi=300)

plt.show()


# %%
both_long = pd.merge(geomid_counts, survey_od, on=["home_geomid", "work_geomid"], suffixes=("_lbs", "_eod"))
intra = both_long[both_long["home_geomid"] == both_long["work_geomid"]]
inter = both_long[both_long["home_geomid"] != both_long["work_geomid"]]

overall_corr = np.log10(both_long['counts_lbs']).corr(np.log10(both_long['counts_eod']))
intra_corr = np.log10(intra['counts_lbs']).corr(np.log10(intra['counts_eod']))
inter_corr = np.log10(inter['counts_lbs']).corr(np.log10(inter['counts_eod']))

# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=intra, x='counts_lbs', y='counts_eod', color="blue", label='Intra-District')
sns.scatterplot(data=inter, x='counts_lbs', y='counts_eod', color="red", label='Inter-District')

min_val = np.min([both_long['counts_lbs'].min(), both_long['counts_eod'].min()])-1
max_val = np.max([both_long['counts_lbs'].max(), both_long['counts_eod'].max()])

identity_line = np.linspace(min_val, max_val, 100)
plt.plot(identity_line, identity_line, 'r--', label='Identity')

plt.xscale('log')
plt.yscale('log')

log_min_val = np.log10(min_val)
log_max_val = np.log10(max_val)

plt.xlim(10**log_min_val, 10**log_max_val)
plt.ylim(10**log_min_val, 10**log_max_val)

plt.ylabel('EOD')
plt.xlabel('LBS')
plt.title('OD Comparison between LBS and EOD - CDMX')

plt.text(0.05, 0.95, f'Overall: r = {overall_corr:.3f}\nIntra: r = {intra_corr:.3f}\nInter: r = {inter_corr:.3f}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend()
plt.savefig("../../../figures/cdmx/district_lbs_survey_correlation_scatter.png", dpi=300)
plt.show()

# %%
