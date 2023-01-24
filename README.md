# Batch Inference Benchmarking with Spark

This repo contains benchmarks for batch inference benchmarking with Spark.

We use the image classification task from the [MLPerf Inference Benchmark suite](https://arxiv.org/pdf/1911.02549.pdf) in the offline setting.
    
- Images from ImageNet 2012 Dataset
- ResNet50 model

The workload is a simple 3 step pipeline:
Read from S3 -> Preprocess images [CPU] -> Predict [GPU]

Images are saved in parquet format (with ~1k images per parquet file). 

We tried with two dataset sizes, 10 GB and 300 GB. These sizes are for when the data is loaded in memory. The compressed on-disk size is much smaller.

We also run a microbenchmark to measure overhead from Spark.

All experiments are run in Databricks using Databricks Runtime v12.0, and using the ML GPU runtime when applicable.

## 10 GB
10 GB dataset using a single-node cluster.

### Configurations

- **Local**: `g4dn.16xlarge` instance (1 GPU). This is the smallest `g4dn` instance that does not OOM.
    - Creates a [single-node cluster](https://docs.databricks.com/clusters/single-node.html) which starts Spark locally on the driver.
    - Local clusters do not support GPU scheduling. Spark will schedule tasks based on available CPU cores.
        - We have to manually repartition the data between the preprocessing and prediction steps to match the number of GPUs.
        - We cannot use multi-GPU machines since we cannot specify the CUDA visible devices for each task.
    - [Code](https://github.com/amogkam/spark-batch-inference-benchmarks/blob/main/code/torch-batch-inference-s3-10G-local.py)
        
- **Standard**. Creates a standard Databricks cluster.
    - This starts a 2 node cluster: 1 node for the driver that does not run tasks, and 1 node for the executor.
    - Standard clusters support GPU scheduling
        - However, since Spark fuses all stages, the effective parallelism is limited by the # of GPUs.
    - Two different instance types:
        - **1 GPU**: `g4dn.xlarge`
        - **4 GPU**: `gd4n.12xlarge`
    - [Code](https://github.com/amogkam/spark-batch-inference-benchmarks/blob/main/code/torch-batch-inference-s3-10G-standard.py)

- **2 stage**. Use 2 separate clusters: 1 CPU-only cluster for preprocessing, and 1 GPU cluster for predicting. We use DBFS to store the intermeditate preprocessed data. This allows preprocessing to scale independently from prediction, at the cost of having to persist data in between the steps.
    - **CPU cluster**: 1 `m6gd.12xlarge` instance with Photon acceleration enabled. This is the smallest `m6gd` instance that does not OOM.
    - **GPU cluster**: 1 `g4dn.12xlarge` instance.
    - [CPU Code](https://github.com/amogkam/spark-batch-inference-benchmarks/blob/main/code/torch-batch-inference-10G-s3-cpu-only.py)
    - [GPU Code](https://github.com/amogkam/spark-batch-inference-benchmarks/blob/main/code/torch-batch-inference-10G-s3-predict-only.py)

### Results

## 300 GB

We pick the best configuration from the 10 GB experiments, and scale up to more nodes for inference on 300 GB data.

4 `g4dn.12xlarge` instances.

### Results

## Microbenchmark



