import os
from typing import Dict
import time

import ray
from ray.data.preprocessors import BatchMapper
from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchCheckpoint, TorchPredictor

import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

BATCH_SIZE = 1000

model = resnet50(weights=ResNet50_Weights.DEFAULT)

os.environ["RAY_DATASET_NEW_EXECUTION_BACKEND"] = "1"

ray.init()

ds = ray.data.read_parquet("s3://air-example-data-2/10G-image-data-synthetic-raw-parquet/")

def preprocess(image_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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

preprocessor = BatchMapper(preprocess, batch_format="numpy")

ckpt = TorchCheckpoint.from_model(model=model, preprocessor=preprocessor)
predictor = BatchPredictor.from_checkpoint(ckpt, TorchPredictor)

start_time = time.time()
predictions = predictor.predict(ds, num_gpus_per_worker=1, feature_columns=["image"], batch_size=BATCH_SIZE, min_scoring_workers=4, max_scoring_workers=4)
predictions.fully_executed()
end_time = time.time()

print(end_time - start_time)
assert predictions.count() == 16232

total_time = end_time - start_time
print(f"Total time: {total_time} seconds")
print(f"Throughput: {predictions.count() / total_time} img/sec")





