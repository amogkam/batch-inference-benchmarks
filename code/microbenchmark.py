# Databricks notebook source
# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

input_data = spark.read.format("parquet").load("s3://air-example-data-2/10G-image-data-synthetic-raw-parquet")
# Force execution of the read
input_data.write.mode("overwrite").format("noop").save()

# COMMAND ----------

# More parallelism than data partitions.
print("# data partitions: ", input_data.rdd.getNumPartitions())
print("# Spark max parallelism: ", sc.defaultParallelism)

# COMMAND ----------

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

import pandas as pd
import time

@pandas_udf(ArrayType(FloatType()))
def dummy_preprocess(image: pd.Series) -> pd.Series:
    time.sleep(1)
    return image

# COMMAND ----------

# Preprocess with a 1 second sleep
# Since the parallelism is more than data partitions, all partitions should run in parallel.
dummy_preprocessed_data = input_data.select(dummy_preprocess(col("image")))

# COMMAND ----------

# Force execution of preprocessing

start_time = time.time()
dummy_preprocessed_data.write.mode("overwrite").format("noop").save()
end_time = time.time()
print(f"Preprocessing took: {end_time-start_time} seconds")

# COMMAND ----------

sc.show_profiles()

# COMMAND ----------

