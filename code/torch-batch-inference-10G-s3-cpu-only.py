# Databricks notebook source
#print("Profiling enabled: ", spark.conf.get("spark.python.profile"))
print("Executor memory: ", spark.conf.get("spark.executor.memory"))

# COMMAND ----------

!pip install numpy -U
!pip install torchvision

# COMMAND ----------

import pandas as pd
from torchvision import transforms
import time

# COMMAND ----------

# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

input_data = spark.read.format("parquet").load("s3://air-example-data-2/10G-image-data-synthetic-raw-parquet")

# COMMAND ----------

# Preprocessing
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

import numpy as np

import torch
import time

# Read documentation here: https://spark.apache.org/docs/3.0.1/sql-pyspark-pandas-with-arrow.html

@pandas_udf(ArrayType(FloatType()))
def preprocess(image: pd.Series) -> pd.Series:
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print(f"number of images: {len(image)}")
    # Spark has no tensor support, so it flattens the image tensor to a single array during read.
    # Each image is represented as a flattened numpy array.
    # We have to reshape back to the original number of dimensions.
    # Need to convert to float dtype otherwise torchvision transforms will complain. The data is read as short (int16) by default
    batch_dim = len(image)
    numpy_batch = np.stack(image.values)
    reshaped_images = numpy_batch.reshape(batch_dim, 256, 256, 3).astype(float)
    
    torch_tensor = torch.Tensor(reshaped_images.transpose(0, 3, 1, 2))
    preprocessed_images = preprocess(torch_tensor).numpy()
    # Arrow only works with single dimension numpy arrays, so need to flatten the array before outputting it
    preprocessed_images = [image.flatten() for image in preprocessed_images]
    return pd.Series(preprocessed_images)

# COMMAND ----------

preprocessed_data = input_data.select(preprocess(col("image")))

# COMMAND ----------

#dbutils.fs.rm("/preprocessed_data/", recurse=True)
dbutils.fs.mkdirs("/preprocessed_data/")
dbutils.fs.ls("/preprocessed_data/")

# COMMAND ----------

start_time = time.time()
preprocessed_data.write.mode("overwrite").format("parquet").save("/preprocessed_data/")
end_time = time.time()
print(f"Preprocessing+Writing took: {end_time-start_time} seconds")
assert preprocessed_data.count() == 16232

# COMMAND ----------

