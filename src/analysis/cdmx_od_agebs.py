#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def round_to_nearest_10(x):
    return np.round(x / 5) * 5

#%%
df = pd.read_csv("../../data/intermediate/od_pairs/cdmx_agebs.csv", dtype={"home_geomid": str, "work_geomid": str})
gdf = gpd.read_file("../../data/intermediate/geometries/cdmx_agebs_zm.geojson")# %%
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

# %%
population.query("expansion < 200").plot(column="expansion", cmap="OrRd", legend=True)

# %%
population.query("expansion < 100").plot(column="population", cmap="OrRd", legend=True)

# %%
population.query("expansion > 1").plot(column="count_uid", cmap="OrRd", legend=True)




# %%
population['scaled_users'] = population['count_uid'] * population['expansion']

# %%
plt.figure(figsize=(10, 10))
sns.scatterplot(data=population.query("count_uid > 5"), x='count_uid', y='population', color="blue")
sns.scatterplot(data=population.query("count_uid > 5"), x='scaled_users', y='population', color="red")
# Add identity line
min_val = min(population.query("count_uid > 5")['count_uid'].min(), population.query("count_uid > 5")['population'].min())
max_val = max(population.query("count_uid > 5")['count_uid'].max(), population.query("count_uid > 5")['population'].max())
identity_line = np.linspace(min_val, max_val, 100)
plt.plot(identity_line, identity_line, 'r--', label='Identity Line')

plt.title('Scaled Population vs. Unique Users')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Total Population')
plt.xlabel('Unique Users')
plt.legend()
plt.show()

# %%
population

# %%
hw_u = df.dropna(subset=["home_geomid", "work_geomid"])#.drop_duplicates(subset=["count_uid"])
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
pivot_df = geomid_counts.pivot(index='home_geomid', columns='work_geomid', values='counts')
pivot_df = pivot_df.reindex(index=full_geomid, columns=full_geomid, fill_value=0).fillna(0)
pivot_df

# %%
# Drop column '0' and row '0' if they exist
if '0' in pivot_df.columns:
    pivot_df = pivot_df.drop(columns=['0'])
if '0' in pivot_df.index:
    pivot_df = pivot_df.drop(index=['0'])
# %%
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_df, cmap='magma', linewidths=0, norm=plt.matplotlib.colors.LogNorm())

plt.title('Heatmap of Home and Work Geomid')
plt.xlabel('Work Geomid')
plt.ylabel('Home Geomid')
plt.show()

# With the survey

# %%
long_df = pivot_df.reset_index().melt(id_vars='home_geomid', var_name='work_geomid', value_name='counts')
both_long = pd.merge(long_df, long_df, on=["home_Distrito", "work_Distrito"], suffixes=("_lbs", "_eod"))
intra = both_long[both_long["home_Distrito"] == both_long["work_Distrito"]]
inter = both_long[both_long["home_Distrito"] != both_long["work_Distrito"]]