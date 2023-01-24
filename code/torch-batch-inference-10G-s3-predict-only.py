# Databricks notebook source
import time
import torch
torch.__version__
torch.cuda.is_available()

# COMMAND ----------

print("Executor memory: ", spark.conf.get("spark.executor.memory"))

# COMMAND ----------

import pandas as pd
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# COMMAND ----------

# Enable Arrow support.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

# Create and broadcast model state. Equivalent to AIR Checkpoint
model_state = resnet50(weights=ResNet50_Weights.DEFAULT).state_dict()
# sc is already initialized by Databricks. Broadcast the model state to all executors.
bc_model_state = sc.broadcast(model_state)

# COMMAND ----------

dbutils.fs.ls("/preprocessed_data/")

# COMMAND ----------

input_data = spark.read.format("parquet").load("/preprocessed_data/")

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")

# COMMAND ----------

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

import numpy as np

import torch

@pandas_udf(ArrayType(FloatType()))
def predict(preprocessed_images: pd.Series) -> pd.Series:
    with torch.inference_mode():
        model = resnet50()
        model.load_state_dict(bc_model_state.value)
        model = model.to(torch.device("cuda")) # Move model to GPU
        model.eval()
        
        batch = preprocessed_images
        batch_dim = len(batch)
        numpy_batch = np.stack(batch.values)
        # Spark has no tensor support, so it flattens the image tensor to a single array during read.
        # Each image is represented as a flattened numpy array.
        # We have to reshape back to the original number of dimensions.
        reshaped_images = numpy_batch.reshape(batch_dim, 3, 224, 224)
        gpu_batch = torch.Tensor(reshaped_images).to(torch.device("cuda"))
        predictions = list(model(gpu_batch).cpu().numpy())
        assert len(predictions) == batch_dim
        
        return pd.Series(predictions)

# COMMAND ----------

predictions = input_data.select(predict(col("preprocess(image)")))

# COMMAND ----------

start_time = time.time()
predictions.write.mode("overwrite").format("noop").save()
end_time = time.time()
print(f"Prediction took: {end_time-start_time} seconds")

assert predictions.count() == 16232

# COMMAND ----------

