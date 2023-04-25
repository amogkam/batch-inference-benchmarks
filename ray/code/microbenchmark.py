import ray
import time

ds = ray.data.read_parquet("s3://air-example-data-2/10G-image-data-synthetic-raw-parquet/")
ds = ds.materialize()

def dummy_preprocess(image):
    time.sleep(1)
    return image

ds = ds.map_batches(dummy_preprocess)

start_time = time.time()
ds.fully_executed()
end_time = time.time()
print(f"Preprocessing took: {end_time-start_time} seconds")