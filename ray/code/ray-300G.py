import ray
from torchvision.models import resnet50, ResNet50_Weights
import torch
import time
from torchvision import transforms

from ray.data import ActorPoolStrategy

BATCH_SIZE = 1000

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model_ref = ray.put(model)

start_time = time.time()
ds = ray.data.read_parquet("s3://air-example-data-2/300G-image-data-synthetic-raw-parquet/")

def preprocess(image_batch):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    torch_tensor = torch.Tensor(image_batch["image"].transpose(0, 3, 1, 2))
    preprocessed_images = preprocess(torch_tensor).numpy()
    return {"image": preprocessed_images}

class Actor:
    def __init__(self, model):
        self.model = ray.get(model)
        self.model.eval()
        self.model.to("cuda")

    def __call__(self, batch):
        with torch.inference_mode():
            output = self.model(torch.as_tensor(batch["image"], device="cuda"))
            return output.cpu().numpy()

start_time_without_metadata_fetching = time.time()
ds = ds.map_batches(preprocess, batch_format="numpy")
ds = ds.map_batches(Actor, batch_size=BATCH_SIZE, compute=ActorPoolStrategy(size=16), num_gpus=1, batch_format="numpy", fn_constructor_kwargs={"model": model_ref}, max_concurrency=2)
for _ in ds.iter_batches(batch_size=None, batch_format="pyarrow"):
    pass
end_time = time.time()

print("Total time: ", end_time-start_time)
print("Throughput (img/sec): ", (488207)/(end_time-start_time))
print("Total time w/o metadata fetching (img/sec) : ", (end_time-start_time_without_metadata_fetching))
print("Throughput w/o metadata fetching (img/sec) ", (488207)/(end_time-start_time_without_metadata_fetching))

print(ds.stats())