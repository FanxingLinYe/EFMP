#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="finetuning_outputsOrig"
DATA_ROOT="/mnt/data0/YXG/NIH_ChestX-ray/"
LOG_DIR="evalOri_logs"
mkdir -p "${LOG_DIR}"

for ckpt in "${CKPT_DIR}"/medsynergyV*_bestauc_checkpoint.bin; do
  base=$(basename "$ckpt")          #  medsynergyV10_bestauc_checkpoint.bin
  ver=${base#medsynergy}            #  V10_bestauc_checkpoint.bin
  ver=${ver%%_*}                    #  V10
  name="NIH_eval_${ver}"

  echo "==> Evaluating $base"
  CUDA_VISIBLE_DEVICES=0 python3 train.py --name "$name" --stage test \
      --model_type ViT-B_16 --model vit_base_patch16 --num_classes 14 \
      --pretrained_path "$ckpt" --eval_batch_size 512 --img_size 224 \
      --dataset_path "$DATA_ROOT" \
  | tee "${LOG_DIR}/${name}.txt"
done
  # CUDA_VISIBLE_DEVICES=0 python3 train.py --name "$name" --stage test \
  #     --model_type ViT-B_16 --model vit_base_patch16 --num_classes 14 \
  #     --pretrained_path "$ckpt" --eval_batch_size 64 --img_size 224 \
  #     --dataset_path "$DATA_ROOT" --data_volume 100 \
  # | tee "${LOG_DIR}/${name}.txt"
