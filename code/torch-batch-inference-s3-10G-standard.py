# Databricks notebook source
import time
import torch
torch.__version__
torch.cuda.is_available()

# COMMAND ----------

#print("Profiling enabled: ", spark.conf.get("spark.python.profile"))
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
    reshaped_images = numpy_batch.reshape(batch_dim, 256, 256, 3).astype(np.float)
    
    torch_tensor = torch.Tensor(reshaped_images.transpose(0, 3, 1, 2))
    preprocessed_images = preprocess(torch_tensor).numpy()
    # Arrow only works with single dimension numpy arrays, so need to flatten the array before outputting it
    preprocessed_images = [image.flatten() for image in preprocessed_images]
    return pd.Series(preprocessed_images)

# COMMAND ----------

preprocessed_data = input_data.select(preprocess(col("image")))

# COMMAND ----------

# 1000 is the largest batch size that can fit on GPU. Limit batch size to 1000 to avoid CUDA OOM.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")

# COMMAND ----------

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

predictions = preprocessed_data.select(predict(col("preprocess(image)")))

# COMMAND ----------

start_time = time.time()
predictions.write.mode("overwrite").format("noop").save()
end_time = time.time()
print(f"Prediction took: {end_time-start_time} seconds")

assert preprocessed_data.count() == 16232

# COMMAND ----------

