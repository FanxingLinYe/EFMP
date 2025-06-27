import torch
import torch.nn as nn
from functools import partial
import timm
assert timm.__version__ == "0.6.12"  # version check
from timm.models.vision_transformer import VisionTransformer

def vit_base_patch16(**kwargs):
    model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# model definition
model = vit_base_patch16(num_classes=14, drop_path_rate=0.1, global_pool="avg")

# load the fine-tuned model
checkpoint_path = "/mnt/data0/YXG/MedSynergy/NIH_ChestX-ray/finetuning_outputsOrig/medsynergyV1_bestauc_checkpoint.bin"
checkpoint_model = torch.load(checkpoint_path, map_location="cpu")  # 直接加载 state_dict
model.load_state_dict(checkpoint_model, strict=False)
