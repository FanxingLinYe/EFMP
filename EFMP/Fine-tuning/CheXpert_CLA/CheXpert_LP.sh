# 数据量 1%
CUDA_VISIBLE_DEVICES=5 python train.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO efmp_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/1/" --data_volume '1' --num_steps 9000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-3 --warmup_steps 150 --fp16 --fp16_opt_level O2 --train_batch_size 768 --mode LinearProbe

# 数据量 10%
CUDA_VISIBLE_DEVICES=5 python train.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO efmp_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/10/" --data_volume '10' --num_steps 9000 --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-2 --warmup_steps 1500 --fp16 --fp16_opt_level O2 --train_batch_size 1024 --mode LinearProbe

# 数据量 100%
CUDA_VISIBLE_DEVICES=5 python train.py --name efmp --stage train --model vit_base_patch16 --task CheXpert --num_classes 5 \
    --pretrained_path '$PATH TO efmp_ViT_Base_16.pth' --dataset_path '$PATH TO CheXpert' \
    --output_dir "output/CheXpert/100/" --data_volume '100' --num_steps 22500 --eval_batch_size 512 --img_size 224 \
    --learning_rate 3e-2 --warmup_steps 3750 --fp16 --fp16_opt_level O2 --train_batch_size 4096 --mode LinearProbe
