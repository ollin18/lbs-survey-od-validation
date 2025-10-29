import pandas as pd
import numpy as np
import os
# Plotting in terminal
import plotext as plt

###################################
###################################
###################################
######## Home #####################
###################################
###################################
###################################
tvivienda = pd.read_csv("../../../../data/raw/cdmx_survey/tvivienda_eod2017/conjunto_de_datos/tvivienda.csv", dtype={"distrito": str, "ent": str, "mun": str})
tvivienda["population"] = tvivienda["p1_1"] * tvivienda["factor"]
tvivienda["muni"] = tvivienda["ent"] + tvivienda["mun"]
tvivienda = tvivienda[["distrito", "muni", "population"]].rename(columns={"distrito": "geomid"})

population = tvivienda.groupby("geomid")["population"].sum().reset_index()

###################################
###################################
###################################
######## Work #####################
###################################
###################################
###################################
tviaje = pd.read_csv("../../../../data/raw/cdmx_survey/tviaje_eod2017/conjunto_de_datos/tviaje.csv", dtype={"dto_origen": str, "dto_dest": str})
# Keep only trips to work
work = tviaje.loc[tviaje["p5_13"] == 2]
work = work[["dto_origen", "dto_dest", "factor"]]
work = work.rename(columns={"dto_origen": "home_geomid", "dto_dest": "work_geomid"})

work = work.loc[work["home_geomid"].isin(population["geomid"].values)]
work = work.loc[work["work_geomid"].isin(population["geomid"].values)]
w_geomid = work.groupby("work_geomid").apply(lambda x: x.loc[np.repeat(x.index.values, x['factor'].astype(int))]).reset_index(drop=True).groupby("work_geomid").count().reset_index()
w_geomid.rename(columns={"home_geomid": "workers"}, inplace=True)
w_geomid.drop(columns=["factor"], inplace=True)


###################################
###################################
###################################
######## OD Matrix ################
###################################
###################################
###################################
unique_geomid = list(set(work["home_geomid"].values) | set(work["work_geomid"].values))
full_geomid = sorted(unique_geomid)

hm = work.copy()
hm = hm.loc[hm.index.repeat(hm['factor'].astype(int))].reset_index(drop=True)

district_counts_od = hm.groupby(['home_geomid', 'work_geomid']).size().reset_index(name='counts')

pivot_df_od = district_counts_od.pivot(index='home_geomid', columns='work_geomid', values='counts')

# Reindex both rows and columns with the full set of districts and fill missing values with 0
pivot_df_od = pivot_df_od.reindex(index=full_geomid, columns=full_geomid, fill_value=0).fillna(0).astype(int)

###################################
###################################
###################################
####### Start Travel Time #########
###################################
###################################
###################################

# Home based work trips by hour of the day
job = tviaje[["id_soc", "p5_13", "p5_6", "p5_9_1", "p5_11a", "p5_3", "factor"]]
job = job.loc[job["p5_9_1"] != 99]
job.rename(columns={"p5_9_1": "start_hour", "p5_6": "type_origin", "p5_13":"propos", "p5_11a":"type_dest"}, inplace=True)

to_work = job.loc[(job["type_origin"] == 1) & (job["propos"].isin([2]))]
back_home = job.loc[(job["type_dest"] == 1) & (job["propos"] == 1)]
hw_trips = pd.merge(to_work, back_home, left_on=["id_soc","type_dest"], right_on=["id_soc","type_origin"], how="inner", suffixes=("_origin", "_destination")).drop_duplicates()

hw_trips = hw_trips.loc[hw_trips.index.repeat(hw_trips['factor_origin'].astype(int))].reset_index(drop=True)
perc_trip = 100*(pd.concat([hw_trips["start_hour_origin"], hw_trips["start_hour_destination"]]).value_counts().sort_index() / (hw_trips.shape[0]*2))

x = perc_trip.index.tolist()
y = perc_trip.values.tolist()
plt.plot(x,y)
plt.title('Percentage of Home-based Work Trips by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Percentage of Trips')
plt.show()

DATA_PATH = "../../../../data/clean/cdmx/survey/"
# Create directory if it doesn't exist
os.makedirs(DATA_PATH, exist_ok=True)
perc_trip.reset_index().rename(columns={"index": "hour", "count": "percentage"}).to_csv(os.path.join(DATA_PATH,"hw_trips_by_hour.csv"), index=False)

# Home based other trips by hour of the day
to_work = job.loc[(job["type_origin"] == 1) & (~job["propos"].isin([2]))]
back_home = job.loc[(job["type_dest"] == 1)]
# hw_trips = pd.merge(to_work, back_home, left_on=["id_soc"], right_on=["id_soc"], how="inner", suffixes=("_origin", "_destination")).drop_duplicates()
hw_trips = pd.merge(to_work, back_home, left_on=["id_soc","type_dest"], right_on=["id_soc","type_origin"], how="inner", suffixes=("_origin", "_destination")).drop_duplicates()

hw_trips = hw_trips.loc[hw_trips.index.repeat(hw_trips['factor_origin'].astype(int))].reset_index(drop=True)
perc_trip = 100*(pd.concat([hw_trips["start_hour_origin"], hw_trips["start_hour_destination"]]).value_counts().sort_index() / (hw_trips.shape[0]*2))

x = perc_trip.index.tolist()
y = perc_trip.values.tolist()
plt.plot(x,y)
plt.title('Percentage of Home-based Other Trips by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Percentage of Trips')
plt.show()

perc_trip.reset_index().rename(columns={"index": "hour", "count": "percentage"}).to_csv(os.path.join(DATA_PATH,"ho_trips_by_hour.csv"), index=False)

# Non-home based trips by hour of the day
hw_trips = job.loc[(job["type_origin"] != 1) & (job["type_dest"] != 1)]
hw_trips = hw_trips.loc[hw_trips.index.repeat(hw_trips['factor'].astype(int))].reset_index(drop=True)
perc_trip = 100*(hw_trips["start_hour"].value_counts().sort_index() / hw_trips.shape[0])

x = perc_trip.index.tolist()
y = perc_trip.values.tolist()
plt.plot(x,y)
plt.title('Percentage of Non-home based trips')
plt.xlabel('Hour of the Day')
plt.ylabel('Percentage of Trips')
plt.show()

perc_trip.reset_index().rename(columns={"index": "hour", "count": "percentage"}).to_csv(os.path.join(DATA_PATH,"nhb_trips_by_hour.csv"), index=False)

# All trips
hw_trips = job.copy()
hw_trips = hw_trips.loc[hw_trips.index.repeat(hw_trips['factor'].astype(int))].reset_index(drop=True)
perc_trip = 100*(hw_trips["start_hour"].value_counts().sort_index() / hw_trips.shape[0])

x = perc_trip.index.tolist()
y = perc_trip.values.tolist()
plt.plot(x,y)
plt.title('Percentage of All Trips by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Percentage of Trips')
plt.show()

perc_trip.reset_index().rename(columns={"index": "hour", "count": "percentage"}).to_csv(os.path.join(DATA_PATH,"all_trips_by_hour.csv"), index=False)

# remove all data from plt
plt.clear_data()
