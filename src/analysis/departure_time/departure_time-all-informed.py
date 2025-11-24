#!/usr/bin/env python
# coding: utf-8
from pyspark.sql import SparkSession, functions as F
import pandas as pd

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, hour
from pyspark.storagelevel import StorageLevel

from pyspark.sql.functions import udf
from pyspark.sql.types import TimestampType
import numpy as np
from datetime import datetime, timedelta



try:
    spark.stop()
except:
    pass

# I wrote this based on a savio2 setup. If more memory or cores are needed,
# adjust here.
spark = (
    SparkSession.builder
    .appName("MobilityAnalysis")
    .master("local[20]")

    .config("spark.driver.memory", "60g")
    .config("spark.driver.maxResultSize", "20g")
    .config("spark.executor.memory", "60g")

    .config("spark.local.dir", "/global/scratch/p2p3/pl1_lbs/ollin") # This is
    #  my scratch space on savio2, please change as needed.

    .config("spark.sql.shuffle.partitions", "40")
    .config("spark.default.parallelism", "40")

    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.3")

    .getOrCreate()
)

print(f"Spark running on: {spark.sparkContext.master}")
print(f"Driver memory: {spark.conf.get('spark.driver.memory')}")


# ---------------------------
# CONFIG
# ---------------------------
country = "MX"

PARQUET_DIR_IN = f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023"

# Column names in your schema
UID_COL              = "uid"
ORDER_COL            = "stop_event"
ORIGIN_END_EPOCH_COL = "end_timestamp"    # epoch seconds (long)
ORIGIN_END_TS_COL    = "end_stop_datetime"  # timestamp (for reference if needed)
NEXT_START_EPOCH_COL = "start_timestamp"  # we'll lead() this
NEXT_START_TS_COL    = "stop_datetime"    # timestamp (for same-day check via to_date)
LOC_TYPE_COL         = "location_type"

# ---------------------------
# LOAD
# ---------------------------
df_clean = spark.read.parquet(PARQUET_DIR_IN)

total_rows = df_clean.count()
print(f"Total rows: {total_rows:,}")

months = df_clean.select("month").distinct().orderBy("month").collect()
print(f"Months to process: {[m['month'] for m in months]}")

hour_dist = pd.read_csv("/global/home/users/ollin/test_cuebiq/informed_versions/all_trips_by_hour.csv")

hour_dist = hour_dist.rename(columns={"start_hour":"hour"})

HBW_WEEKDAY_DIST = dict(zip(hour_dist['hour'], hour_dist['percentage']))


total = sum(HBW_WEEKDAY_DIST.values())
HBW_DIST_NORM = {k: v/total for k, v in HBW_WEEKDAY_DIST.items()}

def sample_departure_time(origin_end_epoch, dest_start_epoch, seed=None):
    """
    Sample departure time from NHTS distribution within valid window
    """
    if origin_end_epoch is None or dest_start_epoch is None:
        return None

    origin_end = datetime.fromtimestamp(origin_end_epoch)
    dest_start = datetime.fromtimestamp(dest_start_epoch)

    valid_hours = []
    valid_probs = []

    current = origin_end.replace(minute=0, second=0, microsecond=0)
    end = dest_start.replace(minute=59, second=59)

    while current <= end:
        hour = current.hour
        if current >= origin_end and current <= dest_start:
            valid_hours.append(hour)
            valid_probs.append(HBW_DIST_NORM.get(hour, 0.01))
        current += timedelta(hours=1)

    if not valid_hours:
        return origin_end_epoch + (dest_start_epoch - origin_end_epoch) * np.random.random()

    prob_sum = sum(valid_probs)
    if prob_sum > 0:
        valid_probs = [p/prob_sum for p in valid_probs]
    else:
        valid_probs = [1.0/len(valid_hours)] * len(valid_hours)

    sampled_hour = np.random.choice(valid_hours, p=valid_probs)

    sampled_dt = origin_end.replace(hour=sampled_hour, minute=0, second=0, microsecond=0)

    minute_offset = np.random.randint(0, 60)
    sampled_dt += timedelta(minutes=minute_offset)

    sampled_epoch = sampled_dt.timestamp()
    if sampled_epoch < origin_end_epoch:
        sampled_epoch = origin_end_epoch
    elif sampled_epoch > dest_start_epoch:
        sampled_epoch = dest_start_epoch

    return sampled_epoch

sample_departure_udf = udf(sample_departure_time, "double")


from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, hour
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import row_number

import os

# ---------------------------
# CONFIG
# ---------------------------
country = "MX"
TEMP_DIR = f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_informed_all/{country}_2023"
spark.sparkContext.setCheckpointDir(f"{TEMP_DIR}/_chkpt")

PARQUET_DIR_IN = f"/global/scratch/p2p3/pl1_lbs/data/quadrant/stops_test/{country}_2023"

UID_COL              = "uid"
ORDER_COL            = "stop_event"
ORIGIN_END_EPOCH_COL = "end_timestamp"     # epoch seconds (long)
NEXT_START_EPOCH_COL = "start_timestamp"   # epoch seconds (long) of the NEXT stop
NEXT_START_TS_COL    = "stop_datetime"
LOC_TYPE_COL         = "location_type"     # 'H', 'W', or other
WEEKEND_COL          = "weekend"           # boolean

# ---------------------------
# LOAD
# ---------------------------
df_clean = spark.read.parquet(PARQUET_DIR_IN)
# df_clean = df_clean.filter(col("location_type") != "H")

total_rows = df_clean.count()
print(f"Total rows: {total_rows:,}")

months = df_clean.select("month").distinct().orderBy("month").collect()
print(f"Months to process: {[m['month'] for m in months]}")

os.makedirs(TEMP_DIR.rsplit('/', 1)[0], exist_ok=True)

# ---------------------------
# LOOP PER MONTH
# ---------------------------
# use your slice or all months
for month_row in months[1:3]:  # replace with "months" to process all
    month = month_row['month']
    print(f"\n{'='*60}\nProcessing month {month}...\n{'='*60}")

    df_month = (
        df_clean
        .filter(col("month") == month)
        .repartition(200)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    print(f"Rows in month {month}: {df_month.count():,}")

    w = Window.partitionBy(UID_COL).orderBy(ORDER_COL)

    df_with_next = (
        df_month
        .withColumn("next_cluster_label", F.lead("cluster_label").over(w))
        .withColumn("dest_start_epoch",   F.lead(col(NEXT_START_EPOCH_COL)).over(w))   # long
        .withColumn("dest_location_type", F.lead(col(LOC_TYPE_COL)).over(w))           # 'H'/'W'/other
        .withColumn("dest_weekend",       F.lead(col(WEEKEND_COL)).over(w))            # boolean
    )

    df_trips = (
        df_with_next
        .filter(
            (col("next_cluster_label").isNotNull()) &
            (col("cluster_label") != col("next_cluster_label")) &
            (col(WEEKEND_COL) == F.lit(False)) &
            (col("dest_weekend") == F.lit(False)) &
            col(ORIGIN_END_EPOCH_COL).isNotNull() &
            col("dest_start_epoch").isNotNull() &
            (col("dest_start_epoch") > col(ORIGIN_END_EPOCH_COL))
        )
        .withColumn("origin_end_date", F.to_date(F.from_unixtime(col(ORIGIN_END_EPOCH_COL))))
        .withColumn("dest_start_date", F.to_date(F.from_unixtime(col("dest_start_epoch"))))
        .filter(col("origin_end_date") == col("dest_start_date"))
        .withColumn(
            "rand_start_epoch",
            sample_departure_udf(col(ORIGIN_END_EPOCH_COL), col("dest_start_epoch"))
        )
        .withColumn("rand_start_ts", F.to_timestamp(F.from_unixtime(col("rand_start_epoch"))))
        .withColumn("trip_start_hour", hour(col("rand_start_ts")))
    )

    w_first_hw = Window.partitionBy(UID_COL, "origin_end_date").orderBy("rand_start_epoch")
    w_last_wh = Window.partitionBy(UID_COL, "origin_end_date").orderBy(col("rand_start_epoch").desc())

    df_trips_slim = (
        df_trips
        .select(UID_COL, "rand_start_ts", "trip_start_hour")
        .withColumn("month", lit(month))
    )

    df_trips_slim = df_trips_slim.persist(StorageLevel.DISK_ONLY)
    trips_in_month = df_trips_slim.count()
    print(f"Same-day WEEKDAY trips (H/W at origin OR destination) in month {month}: {trips_in_month:,}")

    (df_trips_slim
        .repartition(200)
        .write
        .mode("append")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", 5_000_000)
        .partitionBy("month")
        .parquet(f"{TEMP_DIR}/trips"))

    df_trips_slim.unpersist()
    df_month.unpersist()


from pyspark.sql import Window
from pyspark.sql.functions import col, lead, hour, count, lit, round as spark_round
from pyspark.storagelevel import StorageLevel
# TEMP_DIR = f"/global/scratch/p2p3/pl1_lbs/data/quadrant/temp_trips_informed/{country}_2023"

country = "MX"
df_all_trips = spark.read.parquet(f"{TEMP_DIR}/trips")

total_trips = df_all_trips.count()
print(f"\nTotal trips across all months: {total_trips:,}")

trips_by_hour = df_all_trips.groupBy("trip_start_hour") \
    .agg(count("*").alias("trip_count"))

trips_by_hour = trips_by_hour.withColumn(
    "percentage_of_trips",
    spark_round((col("trip_count") / lit(total_trips)) * 100, 2)
)

from pyspark.sql.types import IntegerType

hours_df = spark.createDataFrame([(i,) for i in range(24)], ["trip_start_hour"])
result_complete = hours_df.join(
    trips_by_hour,
    on="trip_start_hour",
    how="left"
).fillna(0).orderBy("trip_start_hour")

print("\nTrips by Hour:")
result_complete.show(24)


result_pandas = result_complete.toPandas()


result_pandas["percentage_of_trips"] = (result_pandas["trip_count"]/result_pandas["trip_count"].sum())*100


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(result_pandas['trip_start_hour'],
         result_pandas['percentage_of_trips'],
         marker='o',
         linewidth=2.5,
         markersize=8,
         color='#2E86AB',
         markerfacecolor='#2E86AB',
         markeredgecolor='white',
         markeredgewidth=1.5)

plt.xlabel('Hour of the Day', fontsize=14, fontweight='bold')
plt.ylabel('Percentage of Trips', fontsize=14, fontweight='bold')
plt.title('Percentage of All Trips by Hour of the Day', fontsize=16, fontweight='bold', pad=20)

plt.xticks(range(0, 24, 1), fontsize=11)
plt.yticks(fontsize=11)
plt.xlim(-0.5, 23.5)
plt.ylim(0, max(result_pandas['percentage_of_trips']) * 1.1)

plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
plt.gca().set_axisbelow(True)

plt.gca().set_facecolor('#F8F9FA')
plt.gcf().patch.set_facecolor('white')

for spine in plt.gca().spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig(f'trips_by_hour_{country}_2023.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as trips_by_hour_{country}_2023.png")
print(f"\nSummary Statistics:")
print(result_pandas.to_string(index=False))


# In[14]:


df_plot = hour_dist.copy()
df_plot['hour'] = df_plot['hour'].astype(int)
df_plot = df_plot.sort_values('hour')


# In[15]:


import numpy as np

# --- Align both series on hours 0..23 just in case ---
hours = np.arange(24)
y_a = np.interp(hours, result_pandas['trip_start_hour'].values,
                result_pandas['percentage_of_trips'].values)
y_b = np.interp(hours, df_plot['hour'].values,
                df_plot['percentage'].values)

# --- Simple DTW (O(nm); fine for 24x24) ---
def dtw_distance(x, y):
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])   # L1 cost; use (xi - y[j-1])**2 for L2
            D[i, j] = cost + min(D[i - 1, j],    # insertion
                                 D[i, j - 1],    # deletion
                                 D[i - 1, j - 1])# match
    return D[n, m]

dtw_val = dtw_distance(y_a, y_b)
# Optional: length-normalized DTW for scale comparability
dtw_norm = dtw_val / (len(y_a) + len(y_b))

print(f"DTW = {dtw_val:.3f} | DTW_norm = {dtw_norm:.3f}")


# In[16]:


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import numpy as np

# --- Ensure dtypes & sort (optional but helps tidy plots) ---
result_pandas = result_pandas.copy()
result_pandas['trip_start_hour'] = result_pandas['trip_start_hour'].astype(int)
result_pandas = result_pandas.sort_values('trip_start_hour')

df_plot = hour_dist.copy()
df_plot['hour'] = df_plot['hour'].astype(int)
df_plot = df_plot.sort_values('hour')

# --- Figure & axes ---
plt.figure(figsize=(15, 8))
ax = plt.gca()

# --- Colors (colorblind-friendly contrast) ---
color_a = '#2E86AB'  # blue for result_pandas
color_b = '#D1495B'  # red-ish for df

# --- Plot lines ---
ax.plot(
    result_pandas['trip_start_hour'],
    result_pandas['percentage_of_trips'],
    marker='o', linewidth=3.5, markersize=9,
    color=color_a, markerfacecolor=color_a,
    markeredgecolor='white', markeredgewidth=1.8,
    label='LBS Data'
)

ax.plot(
    df_plot['hour'],
    df_plot['percentage'],
    marker='s', linewidth=3.5, markersize=9,
    color=color_b, markerfacecolor=color_b,
    markeredgecolor='white', markeredgewidth=1.8,
    label='OD Survery CDMX'
)

# --- Labels & title (large fonts) ---
ax.set_xlabel('Hour of Day', fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel('Percentage of Trips', fontsize=18, fontweight='bold', labelpad=10)
ax.set_title('Percentage of All Trips by Hour of the Day', fontsize=22, fontweight='bold', pad=22)

# --- Axis formatting ---
ax.set_xlim(-0.5, 23.5)
ymax = max(
    np.nanmax(result_pandas['percentage_of_trips']),
    np.nanmax(df_plot['percentage'])
)
ax.set_ylim(0, ymax * 1.12)

ax.set_xticks(range(0, 24, 1))
ax.tick_params(axis='both', labelsize=14)

# Format y-axis as percentages if values are already in percent numbers (e.g., 0–12)
# If yours are actual percents (not 0–1), keep as-is; if they’re proportions (0–1), multiply by 100 first.
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:.0f}%'))

# --- Legend (large & clean) ---
leg = ax.legend(
    loc='upper left', frameon=True, fontsize=14,
    title='Series', title_fontsize=15
)
leg.get_frame().set_edgecolor('#CCCCCC')
leg.get_frame().set_linewidth(1.2)

# --- Grid & aesthetics ---
ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.9)
ax.set_axisbelow(True)
ax.set_facecolor('#FAFAFA')
plt.gcf().patch.set_facecolor('white')

for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')
    spine.set_linewidth(1.3)

handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color='none'))  # dummy handle
labels.append(f"DTW = {dtw_norm:.3f} (normalized)")  # <-- you were missing this line
ax.legend(handles, labels, loc='upper left', frameon=True, fontsize=14,
          title='Series', title_fontsize=15)
plt.tight_layout()

# --- Save & show ---
outfile = f'trips_by_hour_comparison_{country}_2023.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as {outfile}")


# In[ ]:




