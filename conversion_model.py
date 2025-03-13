from huggingface_hub import hf_hub_download
import cv2
import torch
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2
from openvino import convert_model, save_model
from pathlib import Path

encoder = "vits"
model_type = "Small"
model_id = f"depth_anything_v2_{encoder}"

model_path = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_type}", filename=f"{model_id}.pth", repo_type="model")
model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

OV_DEPTH_ANYTHING_PATH = Path(f"{model_id}.xml")

if not OV_DEPTH_ANYTHING_PATH.exists():
    ov_model = convert_model(model, example_input=torch.rand(1, 3, 518, 518), input=[1, 3, 518, 518])
    save_model(ov_model, OV_DEPTH_ANYTHING_PATH)
