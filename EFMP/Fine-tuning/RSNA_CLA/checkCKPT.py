# import timm
# model = timm.models.vit_base_patch16_224(num_classes=14, global_pool="avg", use_fc_norm=False)
# print(model)

# import torch
# from models_vit import vit_base_patch16
# model = vit_base_patch16(num_classes=14, global_pool="avg")
# print(model.cls_token.shape)  # 期望：torch.Size([1, 1, 768])
from utils.data_utils import get_loader
import argparse

args = argparse.Namespace(
    stage="test", dataset_path="/mnt/data0/YXG/NIH_ChestX-ray/", data_volume="100",
    eval_batch_size=64, img_size=224, local_rank=-1
)
test_loader = get_loader(args)
for batch in test_loader:
    x, y = batch
    print(f"Image shape: {x.shape}, Label shape: {y.shape}")
    break
