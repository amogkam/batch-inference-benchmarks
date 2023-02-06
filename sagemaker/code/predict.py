import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet50

import os
import io
from PIL import Image
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def model_fn(model_dir):
    device = torch.device("cuda")
    model = resnet50()
    with open(os.path.join(model_dir, "model.ckpt"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model = model.to(device)
    model.eval()
    return model

# https://stackoverflow.com/questions/62415237/aws-sagemaker-using-parquet-file-for-batch-transform-job
def load_parquet_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    reader = pa.BufferReader(image_as_bytes)
    df = pq.read_table(reader)
    batch_dim = len(df)
    numpy_batch = np.stack(df["image"].to_numpy())
    reshaped_images = numpy_batch.reshape(batch_dim, 256, 256, 3).astype(float)
    torch_tensor = torch.Tensor(reshaped_images.transpose(0, 3, 1, 2))
    
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    preprocessed_images = preprocess(torch_tensor)
    return preprocessed_images


def load_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    image_tensor = preprocess(image_tensor)
    return image_tensor


def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-parquet":
        image_tensor = load_parquet_from_bytearray(request_body)
    elif request_content_type == "application/x-image":
        image_tensor = load_from_bytearray(request_body)
    else:
        raise ValueError("Expected `application/x-parquet`.")
    return image_tensor


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    with torch.inference_mode():
        output = model.forward(input_object.to(torch.device("cuda")))

    return {"predictions": output.cpu().numpy()}