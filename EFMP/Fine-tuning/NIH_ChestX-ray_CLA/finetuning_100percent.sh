
# For 216:
CUDA_VISIBLE_DEVICES=3 python3 train.py --name efmp --stage train --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
    --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/output/checkpoint-119.pth" --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/' \
    --output_dir "output/ChestX-ray14/100/" --data_volume '100' --num_steps 200000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 5000 --fp16 --fp16_opt_level O2 --train_batch_size 96