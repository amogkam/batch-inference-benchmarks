# Batch Inference Benchmarking with SageMaker Batch Transform
SageMaker Batch Transform with 4 `g4dn.xlarge` instances. There is no built-in multi-GPU support, so we cannot use the multi-GPU `g4dn.12xlarge` instance. There are still 4 GPUs total in the cluster.

[Code](sagemaker/code/inference-image.ipynb). Running this code will upload a pre-trained model to S3. It also packages the code in `predict.py` and runs it on the cluster to handle the logic for performing inference.


## Raw Images

SageMaker Batch Transform reads raw images from S3 and sends them as individual HTTP requests to the cluster. Batching across multiple files is not supported

> "SageMaker processes each input file separately. It doesn't combine mini-batches from different input files to comply with the MaxPayloadInMB limit."

https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html.


## Parquet files

We also tried using parquet files 