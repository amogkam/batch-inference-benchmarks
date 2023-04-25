# Batch Inference Benchmarking with SageMaker Batch Transform
SageMaker Batch Transform with 4 `g4dn.xlarge` instances. There is no built-in multi-GPU support, so we cannot use the multi-GPU `g4dn.12xlarge` instance. There are still 4 GPUs total in the cluster.

[Code](sagemaker/code/inference-image.ipynb). Running this code will upload a pre-trained model to S3. It also packages the code in `predict.py` and runs it on the cluster to handle the logic for performing inference.


## Raw Images

SageMaker Batch Transform reads raw images from S3 and sends them as individual HTTP requests to the cluster. Batching across multiple files is not supported

> "SageMaker processes each input file separately. It doesn't combine mini-batches from different input files to comply with the MaxPayloadInMB limit."

https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html.

**Throughput**: 18.702 img/sec
## Parquet files

Since SageMaker does not batch multiple image files together, this means our GPU is extremely underutilized. We tried an additional approach involving batching images together into parquet files beforehand and doing inference on these parquet files.

However, we ran into a few issues:
1. The max payload size is 100 MB, which is far less than the ideal batch size to maximize GPUs
2. It's unclear how to actually parse the input request and read it as a parquet file. Even though code works locally, the job fails with an unhelpful error message, making it impossible to debug:

```
air-example-data-2/10G-image-data-synthetic-raw-parquet-120-partition/644f08f256a24362b744d6219523cd16_000097.parquet: ClientError: 413
```