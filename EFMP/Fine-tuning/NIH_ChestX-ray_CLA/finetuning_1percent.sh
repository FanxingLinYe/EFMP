# ChestX-ray14
# For 216:
# CUDA_VISIBLE_DEVICES=1 python3 train.py --name efmp --stage train --model vit_base_patch16 --model_type ViT-B_16 --num_classes 14 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/output/checkpoint-119.pth" --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/' \
#     --output_dir "output/ChestX-ray14/1/" --data_volume '1' --num_steps 3000 --eval_batch_size 512 --img_size 224 \
#     --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 96

# CUDA_VISIBLE_DEVICES=3 python3 NIH14_FTLP.py --name efmp --stage train --model vit_base_patch16 --task ChestX-ray14 --num_classes 14 \
#     --pretrained_path "/mnt/data0/YXG/MedsARK/efmp/output/checkpoint-115.pth" --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/' \
#     --output_dir "output/ChestX-ray14/1/" --data_volume '1' --num_steps 3000 --eval_batch_size 512 --img_size 224 \
#     --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 96

# CUDA_VISIBLE_DEVICES=1 python train.py --name efmp --stage train --model vit_base_patch16 --task ChestX-ray14 --num_classes 14 \
#     --pretrained_path '/mnt/data0/YXG/MedsARK/efmp/output/checkpoint-119.pth' --dataset_path '/mnt/data0/YXG/NIH_ChestX-ray/images/' \
#     --output_dir "output/ChestX-ray14/1/" --data_volume '1' --num_steps 3000  --eval_batch_size 512 --img_size 224 \
#     --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 96

# RSNA
# For 216:
CUDA_VISIBLE_DEVICES=5 python RSNA_FTLP.py --name efmp --stage train --model vit_base_patch16 --num_classes 1 \
    --pretrained_path '/mnt/data0/YXG/MedsARK/ORDER.pth' --dataset_path '/mnt/data0/YXG/RSNA Pneumonia Detection/' \
    --output_dir "output/RSNA/1/" --data_volume '1' --num_steps 2000  --eval_batch_size 1024 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 256